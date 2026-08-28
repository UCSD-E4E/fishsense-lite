"""Activity to populate the laser-labeling LS project for a dive.

Ports stage 0.3 of `populate_label_studio_project.ipynb` — pushes one
LS task per still-unlabeled image in the dive, then upserts a
LaserLabel row that anchors the (image, LS task, project) triple so
the existing sync workflow (`SyncLabelStudioLaserLabelsWorkflow`) can
pull annotated values back into SQL once a labeler completes them.
"""

from collections import Counter
from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
    publish_label_studio_project,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.object_store import open_object_store_client

PREPROCESS_FOLDER = "preprocess_jpeg"

# Laser LS labeling config (see create_laser_label_studio_project_activity):
# <KeyPointLabels name="laser" toName="img"> with Red/Green Laser labels.
_KEYPOINT_FROM_NAME = "laser"
_KEYPOINT_TO_NAME = "img"
# A keypointlabels result must name a label to be valid, and until 2026-08-28
# that name was always "Red Laser". That is wrong for roughly a quarter of
# prod: of the dives with completed laser labels, 143 are entirely red and 88
# entirely green.
#
# Colour is a property of the rig for a whole dive, not of a frame -- the 31
# dives whose labels are "mixed" carry a 1.2% minority, which is labeler slips
# rather than a laser changing colour mid-dive. So the per-frame colours the
# detector reads off each dot are treated as votes and the dive's majority is
# applied to all of its pre-annotations. Measured over 332 human-labelled
# dots: the per-frame read is 98.48% accurate and the majority got 4 dives of
# 4 right, including one where 10% of frames read the wrong colour.
#
# "Red Laser" remains the fallback when a dive has no colour votes at all --
# it is the more common one, and it is what a labeler saw before this.
_DEFAULT_LASER_LABEL = "Red Laser"
_LASER_LABEL_BY_COLOR = {"red": "Red Laser", "green": "Green Laser"}


def dive_laser_label(predictions) -> str:
    """The LS keypoint label for every pre-annotation in this dive.

    Majority vote over the per-frame colours, so one misread frame cannot
    label its own task differently from the rest of the dive -- a labeler
    seeing both colours inside one dive has no way to tell which to trust.
    Ties and no-votes fall back to the more common colour.
    """
    votes = Counter(
        prediction.color
        for prediction in predictions
        if getattr(prediction, "color", None) in _LASER_LABEL_BY_COLOR
    )
    if not votes:
        return _DEFAULT_LASER_LABEL
    if votes.get("red", 0) == votes.get("green", 0):
        return _DEFAULT_LASER_LABEL
    winner, _ = votes.most_common(1)[0]
    return _LASER_LABEL_BY_COLOR[winner]


def _prediction_annotations(
    prediction, laser_label: str = _DEFAULT_LASER_LABEL
) -> list:
    """Build the LS `predictions` list (one keypoint pre-annotation) from a
    model `LaserPrediction`, or [] when there's nothing placeable.

    Label Studio keypoints are in percentages, so convert the prediction's
    rectified pixels using its own recorded frame dims. Skips a prediction
    with no detection (x/y None) or missing dims.
    """
    if prediction is None:
        return []
    x, y, width, height = (
        prediction.x,
        prediction.y,
        prediction.width,
        prediction.height,
    )
    if x is None or y is None or not width or not height:
        return []
    return [
        {
            "model_version": "laser-detector",
            "result": [
                {
                    "from_name": _KEYPOINT_FROM_NAME,
                    "to_name": _KEYPOINT_TO_NAME,
                    "type": "keypointlabels",
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "x": x / width * 100,
                        "y": y / height * 100,
                        "width": 0.5,
                        "keypointlabels": [laser_label],
                    },
                }
            ],
        }
    ]


def _select_unlabeled_images(
    images: List[Image],
    existing_labels: List[LaserLabel],
    predicted_image_ids: set,
) -> List[Image]:
    """Return images that need a fresh laser LS task: no completed
    LaserLabel in any project, AND a model prediction already exists.

    **Prediction-gated** (like species populate is JPEG-gated): seeding a
    LaserLabel for an un-predicted image would exclude it from the predict
    cohort (which requires "no LaserLabel") before the detector ever ran,
    permanently starving it of a prediction. So an image is only populated
    once its `LaserPrediction` is in place — un-predicted images defer to a
    later run.

    Multi-row-aware: an image carrying a completed row in one project and an
    incomplete sentinel elsewhere is treated as labeled.
    """
    completed_image_ids = {
        label.image_id for label in existing_labels if label.completed
    }
    return [
        image
        for image in images
        if image.id not in completed_image_ids and image.id in predicted_image_ids
    ]


async def _gate_on_jpeg_presence(images: List[Image]) -> List[Image]:
    """Keep only images whose stage-0.1 laser JPEG is already in Garage.

    Populate runs decoupled from preprocess (its own schedule) and is
    prediction-gated — but the predictor works off the raw `.ORF`, so it can
    predict an image whose `preprocess_jpeg` JPEG was never written (e.g. a
    dive labeled before the Garage migration, whose originals live on the old
    file-exchange). Seeding a task for such an image points its LS `data.image`
    at a missing key and the labeler gets a Garage `NoSuchKey`. Gate on JPEG
    existence so those images defer to a later run (once preprocess has written
    them); a no-op when the JPEG is already present. Mirrors species populate's
    `_gate_on_jpeg_presence`.
    """
    if not images:
        return images
    store = open_object_store_client()
    present: List[Image] = []
    for image in images:
        if await store.has_processed_jpeg(PREPROCESS_FOLDER, image.checksum):
            present.append(image)
        else:
            activity.logger.info(
                "laser JPEG not yet in Garage for image %d (checksum=%s); "
                "deferring to a later populate run",
                image.id,
                image.checksum,
            )
    return present


def _build_task(
    image: Image, prediction=None, laser_label: str = _DEFAULT_LASER_LABEL
) -> dict:
    """Build an LS task referencing the preprocessed JPEG.

    Emits BOTH `image` and `img` keys in `data` because legacy prod
    LS projects' labeling-config XML uses different conventions —
    older projects bind `<Image value="$img"/>`, newer ones use
    `value="$image"`. LS's `import_tasks` rejects a payload outright
    if its labeling config's required key is missing (HTTP 400
    "img key is expected in task data"), but extra keys are inert.
    Dual-emission lets `populate` fan out across both shapes without
    interrogating each project's `label_config` first.
    """
    url = build_image_url(PREPROCESS_FOLDER, image.checksum)
    return {
        "data": {"image": url, "img": url},
        "annotations": [],
        # Model-assisted labeling: seed the laser-detector's predicted dot as
        # a pre-annotation the labeler confirms/nudges. Empty when there's no
        # prediction for this image yet.
        "predictions": _prediction_annotations(prediction, laser_label),
    }


@activity.defn
async def populate_laser_label_studio_project_activity(
    dive_id: int, project_id: int
) -> int:
    """Push tasks for every incomplete image in `dive_id` to `project_id`.

    Returns the number of tasks imported (== rows upserted).
    """
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []
        # Canonical frames only. The same physical frames live under several
        # dive rows (half of prod's image table is duplicate content), and
        # `is_canonical` marks which copy is the real one. The cohort selectors
        # gate on it, and CLAUDE.md requires resolvers to mirror the selector
        # predicate exactly -- otherwise the dispatched per-image work would not
        # match what the cohort promised, and the dive could never drain.
        # This also covers on-demand/backfill runs, which bypass the cohort.
        images = [image for image in images if image.is_canonical]
        existing_labels = await fs.labels.get_laser_labels(dive_id) or []
        predictions = await fs.labels.get_laser_predictions(dive_id) or []
        predictions_by_image = {p.image_id: p for p in predictions}
        # One colour for the whole dive — see `dive_laser_label`. Computed over
        # every prediction the dive has, not just the images being seeded now,
        # so a later top-up run agrees with the first.
        laser_label = dive_laser_label(predictions)
        activity.logger.info(
            "dive_id=%d laser colour %s from %d/%d predictions with a reading",
            dive_id,
            laser_label,
            sum(1 for p in predictions if getattr(p, "color", None)),
            len(predictions),
        )

        unlabeled = _select_unlabeled_images(
            images, existing_labels, set(predictions_by_image)
        )
        # Never seed a task for an image whose laser JPEG isn't in Garage —
        # its LS task would 404 (NoSuchKey). Deferred images retry next run.
        unlabeled = await _gate_on_jpeg_presence(unlabeled)
        if not unlabeled:
            activity.logger.info(
                "Dive %d has a completed laser label for every image; "
                "nothing to import",
                dive_id,
            )
            # No image needs a task, so the task set is complete. Publish
            # iff this project already holds tasks — a genuinely empty
            # project (grandfathered dive whose rows point at an old
            # project) is left as a hidden draft.
            if any(
                label.label_studio_project_id == project_id for label in existing_labels
            ):
                await publish_label_studio_project(project_id)
            return 0

        tasks = [
            _build_task(image, predictions_by_image.get(image.id), laser_label)
            for image in unlabeled
        ]

        async def _record(image: Image, task_id: int) -> None:
            label = LaserLabel(
                id=None,
                image_id=image.id,
                label_studio_task_id=task_id,
                label_studio_project_id=project_id,
                updated_at=None,
                completed=False,
                label_studio_json={},
                user_id=None,
                superseded=False,
                x=None,
                y=None,
                label=None,
            )
            await fs.labels.put_laser_label(image.id, label)

        imported = await import_tasks_and_record_labels(
            project_id=project_id,
            tasks=tasks,
            record_label=_record,
            items=unlabeled,
        )
        # Laser imports its whole selection in one pass (no JPEG deferral),
        # so the project's task set is now complete — safe to publish.
        await publish_label_studio_project(project_id)
        return imported

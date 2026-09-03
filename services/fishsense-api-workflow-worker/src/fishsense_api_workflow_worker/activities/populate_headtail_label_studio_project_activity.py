"""Activity to populate the headtail-labeling LS project for a dive.

Source flipped on 2026-05-04 from species top-3 → valid laser labels.
A laser label is "valid" when `completed=True`, `superseded=False`,
and both `x` and `y` are populated — same gate
`perform_laser_calibration_activity` and the validator's
`_positive_xy` already use as "usable laser." Cascading from lasers
lets head/tail labeling kick off as soon as laser labelers (and the
validator) sign off, instead of waiting for the species pass to flag
top-3 measurable angles.

Per-image filter:
  1. Image must carry a valid laser label (gate above).
  2. Image must NOT have a non-sentinel HeadTailLabel row already —
     the existing `id is None or project_id is None` distinction is
     handled implicitly by `_select_target_images` (drops rows with
     `completed=True` only; the "no row at all" idempotency comes
     from the cohort selector).

After importing tasks the activity also marks any pre-existing
incomplete headtail labels for the dive as `superseded=True`, so the
sync workflow's downstream consumers (calibration, measurement) can
ignore stale rows that were obsoleted by a re-import.
"""

from typing import List

from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_shared.headtail_predictor import headtail_model_version_tag
from fishsense_shared.object_store import HEADTAIL_JPEG_FOLDER
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
    publish_label_studio_project,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.object_store import open_object_store_client

# Physical Garage prefix the data-worker writes head/tail JPEGs to (stage 5.1).
# Re-exported from the shared key contract rather than spelled again: the
# predict stage now *reads* this same folder, and two copies of the string are
# two chances for a task URI and an inference to disagree about which image is
# being labelled.
HEADTAIL_FOLDER = HEADTAIL_JPEG_FOLDER


def _is_valid_laser(label: LaserLabel) -> bool:
    """Same predicate the API SQL uses for the cohort gate."""
    return bool(
        label.completed
        and not label.superseded
        and label.x is not None
        and label.y is not None
    )


def _select_target_images(
    laser_labels: List[LaserLabel],
    images_by_id: dict[int, Image],
    existing_headtail_labels: List[HeadTailLabel],
) -> List[Image]:
    """Pick the images that need a fresh headtail LS task.

    Source: laser labels passing `_is_valid_laser`.
    Filter: drop any image whose existing headtail label is already
    completed (so re-running is a no-op for finished work).
    """
    completed_ids = {
        label.image_id for label in existing_headtail_labels if label.completed
    }
    selected: List[Image] = []
    for label in laser_labels:
        if not _is_valid_laser(label):
            continue
        if label.image_id in completed_ids:
            continue
        image = images_by_id.get(label.image_id)
        if image is not None:
            selected.append(image)
    return selected


async def _gate_on_jpeg_presence(images: List[Image]) -> List[Image]:
    """Keep only images whose stage-5.1 head/tail JPEG is already in Garage.

    A task whose `s3://` URI has no object behind it shows the labeler a
    missing image — and because recording it writes a non-sentinel row, the
    dive also drops out of the stage-5.1 cohort, so the JPEG can never be
    rendered afterwards. Prod dive 84 wedged exactly that way on 2026-08-04
    (36 of 39 tasks pointed at nothing) after populate was run standalone.
    Deferring costs nothing: the image is picked up on a later run once
    preprocess has written it, and this is a no-op when populate is chained
    right after preprocess. Mirrors species populate's gate.
    """
    if not images:
        return images
    store = open_object_store_client()
    present: List[Image] = []
    for image in images:
        if await store.has_processed_jpeg(HEADTAIL_FOLDER, image.checksum):
            present.append(image)
        else:
            activity.logger.info(
                "headtail JPEG not yet in Garage for image %d (checksum=%s); "
                "deferring to a later populate run",
                image.id,
                image.checksum,
            )
        activity.heartbeat()
    return present


# The head/tail project's labeling config declares `KeyPointLabels name="kp-1"`
# with `Snout` and `Fork`. Both literals must match it exactly:
# `sync_headtail_labels_for_label_studio_project_activity` filters annotations
# on `from_name == "kp-1"` and picks the two points out by label, so a mismatch
# seeds tasks whose labels never come back.
_KEYPOINT_FROM_NAME = "kp-1"
_KEYPOINT_TO_NAME = "image"
_SNOUT_LABEL = "Snout"
_FORK_LABEL = "Fork"

# A real fish silhouette's mask-area / length**2 runs ~0.15-0.30. Applied here,
# at seed time, rather than in the predictor: every prediction is stored either
# way, so the band can be retuned against rows already collected instead of
# requiring a re-predict. Measured effect on the corpus: keeping 0.18-0.32 drops
# ~24% of predictions and moves p90 length error from 17.1% to 12.7%.
_MIN_SILHOUETTE_RATIO = 0.18
_MAX_SILHOUETTE_RATIO = 0.32


def select_predicted_image_ids(predictions) -> set:
    """Image ids the detector has already visited.

    Populate is **prediction-gated**, for the same reason the laser populate is:
    it seeds sentinel `HeadTailLabel` rows, and the predict cohort requires "no
    live label" — so populating an image before the detector ran would remove
    it from that cohort permanently and starve it of a prediction forever.

    An abstention counts as visited. The image *was* looked at, and holding it
    back from populate on that basis would strand it in the opposite direction:
    never predicted usefully, and never labelled by a human either.
    """
    return {p.image_id for p in predictions}


def prediction_annotations(prediction) -> list:
    """Build the LS `predictions` list — two keypoints — from a model
    `HeadTailPrediction`, or [] when there is nothing placeable.

    Label Studio keypoints are percentages, so the stored rectified pixels are
    converted with the prediction's own recorded frame dims. Returning [] still
    creates the task; it just arrives unseeded, which is the right outcome for
    an abstention, a low-confidence shape, or a row with no dims to convert by.
    """
    if prediction is None:
        return []
    if getattr(prediction, "rejected_low_confidence", False):
        return []
    if getattr(prediction, "status", None) != "predicted":
        return []

    head_x, head_y = prediction.head_x, prediction.head_y
    tail_x, tail_y = prediction.tail_x, prediction.tail_y
    width, height = prediction.width, prediction.height
    if None in (head_x, head_y, tail_x, tail_y) or not width or not height:
        return []

    ratio = getattr(prediction, "silhouette_ratio", None)
    # None means "not recorded", not "out of band" — it must not suppress every
    # row written before the column existed.
    if (
        ratio is not None
        and not _MIN_SILHOUETTE_RATIO <= ratio <= _MAX_SILHOUETTE_RATIO
    ):
        return []

    def _point(x, y, label):
        return {
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
                "keypointlabels": [label],
            },
        }

    return [
        {
            "model_version": headtail_model_version_tag(
                getattr(prediction, "checkpoint", None),
                getattr(prediction, "core_version", None),
            ),
            "result": [
                _point(head_x, head_y, _SNOUT_LABEL),
                _point(tail_x, tail_y, _FORK_LABEL),
            ],
        }
    ]


def _build_task(image: Image, prediction=None) -> dict:
    """Build an LS task. Emits both `image` and `img` keys to satisfy
    legacy LS labeling-config XML across prod projects — see
    `populate_laser_label_studio_project_activity._build_task`."""
    url = build_image_url(HEADTAIL_FOLDER, image.checksum)
    return {
        "data": {"image": url, "img": url},
        "predictions": prediction_annotations(prediction),
        "annotations": [],
    }


@activity.defn
async def populate_headtail_label_studio_project_activity(
    dive_id: int, project_id: int
) -> int:
    """Push headtail tasks for `dive_id` and supersede stale rows.

    Returns the number of tasks imported.
    """
    async with get_fs_client() as fs:
        laser_labels = await fs.labels.get_laser_labels(dive_id) or []
        existing_headtail = await fs.labels.get_headtail_labels(dive_id) or []
        predictions = await fs.labels.get_headtail_predictions(dive_id) or []
        predictions_by_image = {p.image_id: p for p in predictions}
        predicted_ids = select_predicted_image_ids(predictions)

        # Hydrate Image rows by id for the laser-valid candidates.
        target_image_ids = {
            label.image_id for label in laser_labels if _is_valid_laser(label)
        }
        images_by_id: dict[int, Image] = {}
        for image_id in target_image_ids:
            image = await fs.images.get(image_id=image_id)
            if image is not None:
                images_by_id[image.id] = image
            activity.heartbeat()

        targets = _select_target_images(laser_labels, images_by_id, existing_headtail)
        # Prediction-gated: seeding a sentinel row for an unpredicted image
        # would drop it out of the predict cohort before the detector ever ran.
        # See `select_predicted_image_ids`.
        targets = [image for image in targets if image.id in predicted_ids]
        # Never seed a task for an image the data-worker hasn't rendered.
        targets = await _gate_on_jpeg_presence(targets)

        new_count = 0
        if targets:
            tasks = [
                _build_task(image, predictions_by_image.get(image.id))
                for image in targets
            ]

            async def _record(image: Image, task_id: int) -> None:
                label = HeadTailLabel(
                    id=None,
                    image_id=image.id,
                    label_studio_task_id=task_id,
                    label_studio_project_id=project_id,
                    head_x=None,
                    head_y=None,
                    tail_x=None,
                    tail_y=None,
                    updated_at=None,
                    superseded=False,
                    completed=False,
                    label_studio_json={},
                    user_id=None,
                )
                await fs.labels.put_headtail_label(image.id, label)

            new_count = await import_tasks_and_record_labels(
                project_id=project_id,
                tasks=tasks,
                record_label=_record,
                items=targets,
            )
        else:
            activity.logger.info(
                "Dive %d has no laser-valid images needing headtail "
                "labels; skipping task import",
                dive_id,
            )

        refreshed_image_ids = {image.id for image in targets}

        # Supersede pass: dead-letter incomplete rows that this project no
        # longer owns, so its own rows are canonical. Two kinds qualify:
        #   * rows in a DIFFERENT (legacy, pre-per-dive) project, and
        #   * rows in THIS project whose image is no longer a target (its
        #     laser stopped being valid), which are stale tasks to collect.
        # Only acts on rows with an `id` (already persisted).
        #
        # The exemption is "same project AND refreshed this run", and needs
        # both halves:
        #
        #   refreshed — superseding a row this run just re-imported undoes the
        #     run's own work, and because the pass reads a snapshot taken
        #     BEFORE the import and skips already-superseded rows, the outcome
        #     ALTERNATES run to run: live -> superseded -> live... Prod dive
        #     341 oscillated that way on every hourly firing and activity
        #     retry, so its images never held a usable pending row and the
        #     dive never drained from the cohort.
        #
        #   same project — `get_headtail_labels(dive_id)` returns every
        #     non-superseded row for the dive across ALL projects, and
        #     `put_headtail_label` upserts on `(image_id, label_studio_project_id)`, so
        #     ONE IMAGE CAN HOLD TWO ROWS. Exempting by image alone let a
        #     legacy row on a refreshed image survive forever; since
        #     `dive_pipeline_status`'s `*_labeling_complete` needs zero
        #     incomplete non-superseded rows, the dive read incomplete on the
        #     dashboard permanently. (An earlier comment here claimed the
        #     upsert keyed on `image_id` alone, "so there is only ever one row
        #     per image". That was never true.)
        #
        # Species populate exempts by project only — it has no same-project GC
        # because its cohort can't drop an image mid-flight the way a laser
        # revalidation can.
        for old in existing_headtail:
            if old.completed or old.superseded or old.id is None:
                continue
            if (
                old.label_studio_project_id == project_id
                and old.image_id in refreshed_image_ids
            ):
                continue
            old.superseded = True
            await fs.labels.put_headtail_label(old.image_id, old)
            activity.heartbeat()

        # Headtail imports its whole selection in one pass (no JPEG
        # deferral), so the project's task set is complete. Publish iff it
        # actually holds tasks so an empty project isn't shown to labelers.
        if new_count > 0 or any(
            label.label_studio_project_id == project_id for label in existing_headtail
        ):
            await publish_label_studio_project(project_id)

        return new_count

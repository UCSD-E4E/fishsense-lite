"""Activity to populate the dive-slate-labeling LS project for a dive.

Ports stage 11 of `populate_label_studio_project.ipynb`. Slate
candidates are species labels with
`content_of_image == 'Slate, Laser on slate'` — that classification is
made by the species labeler in stage 4. The notebook didn't filter
by existing slate-label completion (re-runs would have duplicated
tasks); this version filters out images that already have a completed
slate label so the workflow is idempotent.
"""

from typing import List

from botocore.exceptions import BotoCoreError, ClientError
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.slate_prediction import SlatePrediction
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
    publish_label_studio_project,
)
from fishsense_api_workflow_worker.activities.sync_dive_slate_labels_for_label_studio_project_activity import (  # noqa: E501  pylint: disable=line-too-long
    compute_pdf_panel_aspect_ratio,
    compute_pdf_panel_width_in_composite,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.object_store import open_object_store_client

# Physical Garage prefix the data-worker writes slate JPEGs to (stage 9).
# Was the nginx virtual name "dive_slate_jpgs"; now the real key prefix.
DIVE_SLATE_FOLDER = "preprocess_slate_images_jpeg"
SLATE_CONTENT_MARKER = "Slate, Laser on slate"

# LS keypoint control names on the dive-slate labeling config (see
# create_dive_slate_label_studio_project_activity's XML) + the sync parser.
_KEYPOINT_FROM_NAME = "reference_points"
_KEYPOINT_TO_NAME = "image"
_REFERENCE_POINT_LABEL = "Reference Point"

# LS `model_version` tag stamped on every slate-detector pre-annotation. Shared
# with the backfill activity so its idempotency check (skip tasks that already
# carry this version) stays in lockstep with what we write.
SLATE_DETECTOR_MODEL_VERSION = "slate-detector"


def _prediction_annotations(prediction, panel_width: float) -> list:
    """Build the LS `predictions` list (one keypoint per reference point) from a
    model `SlatePrediction`, or [] when there's nothing seedable.

    The prediction's `reference_points` are in rectified-photo pixels; the LS
    canvas is the composite (PDF panel left + photo right), so shift x right by
    the rendered `panel_width` and express both as percentages of the composite
    dimensions — the inverse of the panel-offset strip the sync activity applies
    on write-back.
    """
    if prediction is None or not prediction.reference_points:
        return []
    if not prediction.width or not prediction.height:
        return []
    composite_width = float(panel_width) + float(prediction.width)
    composite_height = float(prediction.height)
    result = [
        {
            "from_name": _KEYPOINT_FROM_NAME,
            "to_name": _KEYPOINT_TO_NAME,
            "type": "keypointlabels",
            "original_width": int(round(composite_width)),
            "original_height": int(round(composite_height)),
            "value": {
                "x": (float(x) + float(panel_width)) / composite_width * 100.0,
                "y": float(y) / composite_height * 100.0,
                "width": 0.5,
                "keypointlabels": [_REFERENCE_POINT_LABEL],
            },
        }
        for x, y in prediction.reference_points
    ]
    return [
        {
            "model_version": SLATE_DETECTOR_MODEL_VERSION,
            "score": float(prediction.confidence),
            "result": result,
        }
    ]


async def _slate_panel_aspect(dive_id: int, fs) -> float | None:
    """Fetch the dive's slate PDF from Garage and return its width/height aspect
    (points), or None when it can't be resolved. A missing aspect just means no
    pre-annotation (labeler places from scratch) — never fail populate over it,
    unlike the sync activity where wrong-space *persistence* is the danger.
    """
    dive = await fs.dives.get(dive_id=dive_id)
    slate_id = getattr(dive, "dive_slate_id", None) if dive is not None else None
    if slate_id is None:
        return None
    try:
        pdf_bytes = await open_object_store_client().download_slate_pdf(slate_id)
    except (ClientError, BotoCoreError) as exc:
        activity.logger.warning(
            "slate PDF for slate_id=%s unavailable (%s); seeding slate tasks "
            "without pre-annotations",
            slate_id,
            exc,
        )
        return None
    return compute_pdf_panel_aspect_ratio(pdf_bytes)


def _select_target_image_ids(
    species_labels: List[SpeciesLabel],
    existing_slate_labels: List[DiveSlateLabel],
) -> List[int]:
    """Pick image IDs that need a fresh slate LS task: species-marked
    as slate, with no completed slate label yet."""
    completed_ids = {
        label.image_id for label in existing_slate_labels if label.completed
    }
    return [
        label.image_id
        for label in species_labels
        if label.content_of_image == SLATE_CONTENT_MARKER
        and label.image_id not in completed_ids
    ]


async def _gate_on_jpeg_presence(images: List[Image]) -> List[Image]:
    """Keep only images whose stage-9 slate JPEG is already in Garage.

    A task whose `s3://` URI has no object behind it shows the labeler a
    missing image, and recording the row also drops the dive out of the
    stage-9 cohort so the JPEG can never be rendered afterwards. Deferring
    costs nothing — the image returns on a later run. Mirrors species and
    headtail populate; see the dive-84 incident (2026-08-04).
    """
    if not images:
        return images
    store = open_object_store_client()
    present: List[Image] = []
    for image in images:
        if await store.has_processed_jpeg(DIVE_SLATE_FOLDER, image.checksum):
            present.append(image)
        else:
            activity.logger.info(
                "slate JPEG not yet in Garage for image %d (checksum=%s); "
                "deferring to a later populate run",
                image.id,
                image.checksum,
            )
        activity.heartbeat()
    return present


def _build_task(image: Image, prediction=None, panel_width: float = 0.0) -> dict:
    """Build an LS task. Emits both `image` and `img` keys to satisfy legacy LS
    labeling-config XML across prod projects. Seeds a keypoint pre-annotation
    from the model `SlatePrediction` when one exists (assisted review) — a
    labeler confirms/nudges the board points rather than placing all of them."""
    url = build_image_url(DIVE_SLATE_FOLDER, image.checksum)
    return {
        "data": {"image": url, "img": url},
        "predictions": _prediction_annotations(prediction, panel_width),
        "annotations": [],
    }


@activity.defn
async def populate_dive_slate_label_studio_project_activity(
    dive_id: int, project_id: int
) -> int:
    """Push slate tasks for `dive_id` to `project_id`.

    Returns the number of tasks imported.
    """
    async with get_fs_client() as fs:
        species_labels = await fs.labels.get_species_labels(dive_id) or []
        existing_slate = await fs.labels.get_dive_slate_labels(dive_id) or []

        target_ids = _select_target_image_ids(species_labels, existing_slate)

        # Model pre-annotations (assisted review). The predict parent (+35 min)
        # writes SlatePrediction rows before this populate runs (stage-9 +45);
        # a missing prediction just means no pre-annotation for that frame.
        predictions: List[SlatePrediction] = (
            await fs.labels.get_slate_predictions(dive_id) or []
        )
        prediction_by_image = {p.image_id: p for p in predictions}
        aspect = await _slate_panel_aspect(dive_id, fs) if prediction_by_image else None

        new_count = 0
        images: List[Image] = []
        for image_id in target_ids:
            image = await fs.images.get(image_id=image_id)
            if image is not None:
                images.append(image)
            activity.heartbeat()

        images = await _gate_on_jpeg_presence(images)

        if images:
            def _task(image: Image) -> dict:
                prediction = prediction_by_image.get(image.id)
                panel_width = (
                    compute_pdf_panel_width_in_composite(aspect, prediction.height)
                    if aspect is not None
                    and prediction is not None
                    and prediction.height
                    else 0.0
                )
                return _build_task(image, prediction, panel_width)

            tasks = [_task(image) for image in images]

            async def _record(image: Image, task_id: int) -> None:
                label = DiveSlateLabel(
                    id=None,
                    image_id=image.id,
                    label_studio_task_id=task_id,
                    label_studio_project_id=project_id,
                    image_url=build_image_url(DIVE_SLATE_FOLDER, image.checksum),
                    upside_down=None,
                    reference_points=None,
                    slate_rectangle=None,
                    skipped_points=None,
                    updated_at=None,
                    completed=False,
                    superseded=False,
                    label_studio_json={},
                    user_id=None,
                )
                await fs.labels.put_dive_slate_label(image.id, label)

            new_count = await import_tasks_and_record_labels(
                project_id=project_id,
                tasks=tasks,
                record_label=_record,
                items=images,
            )
        else:
            activity.logger.info(
                "Dive %d has no slate-marked images needing labels; skipping",
                dive_id,
            )

        refreshed_image_ids = {image.id for image in images}

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
        #   same project — `get_dive_slate_labels(dive_id)` returns every
        #     non-superseded row for the dive across ALL projects, and
        #     `put_dive_slate_label` upserts on `(image_id, label_studio_project_id)`, so
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
        for old in existing_slate:
            if old.completed or old.superseded or old.id is None:
                continue
            if (
                old.label_studio_project_id == project_id
                and old.image_id in refreshed_image_ids
            ):
                continue
            old.superseded = True
            await fs.labels.put_dive_slate_label(old.image_id, old)
            activity.heartbeat()

        # Slate imports its whole selection in one pass (no JPEG deferral),
        # so the project's task set is complete. Publish iff it actually
        # holds tasks so an empty project isn't shown to labelers.
        if new_count > 0 or any(
            label.label_studio_project_id == project_id
            for label in existing_slate
        ):
            await publish_label_studio_project(project_id)

        return new_count

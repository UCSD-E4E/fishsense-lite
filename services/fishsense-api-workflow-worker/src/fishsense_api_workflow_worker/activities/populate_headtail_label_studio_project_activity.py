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
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client

HEADTAIL_FOLDER = "headtail_jpeg"


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


def _build_task(image: Image) -> dict:
    """Build an LS task. Emits both `image` and `img` keys to satisfy
    legacy LS labeling-config XML across prod projects — see
    `populate_laser_label_studio_project_activity._build_task`."""
    url = build_image_url(HEADTAIL_FOLDER, image.checksum)
    return {
        "data": {"image": url, "img": url},
        "predictions": [],
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

        # Hydrate Image rows by id for the laser-valid candidates.
        target_image_ids = {
            label.image_id for label in laser_labels if _is_valid_laser(label)
        }
        images_by_id: dict[int, Image] = {}
        for image_id in target_image_ids:
            image = await fs.images.get(image_id=image_id)
            if image is not None:
                images_by_id[image.id] = image

        targets = _select_target_images(
            laser_labels, images_by_id, existing_headtail
        )

        new_count = 0
        if targets:
            tasks = [_build_task(image) for image in targets]

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

        # Supersede pass: mark all previously-incomplete headtail labels
        # for this dive as superseded so newly-created rows are canonical.
        # Mirrors the notebook's behavior. Only acts on rows with an `id`
        # (i.e. already-persisted) — newly-inserted rows from the import
        # block above don't yet have one in this scope.
        for old in existing_headtail:
            if old.completed or old.superseded or old.id is None:
                continue
            old.superseded = True
            await fs.labels.put_headtail_label(old.image_id, old)
            activity.heartbeat()

        return new_count

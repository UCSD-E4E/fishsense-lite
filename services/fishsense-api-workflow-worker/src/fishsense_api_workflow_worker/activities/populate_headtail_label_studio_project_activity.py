"""Activity to populate the headtail-labeling LS project for a dive.

Ports stage 5.3 of `populate_label_studio_project.ipynb`. The notebook
swept all HIGH-priority canonical dives in one pass; this version
takes a single `dive_id` to match how the operator labels day-to-day
(dive 0.3 -> 4 -> 5.3 -> 11 sequentially per dive). The all-dive
sweep can be added as a higher-level workflow that fans out across
canonical dives if/when batch operation is needed.

Per the notebook, this stage filters source images two ways:
  1. Only species labels with `top_three_photos_of_group == True` are
     candidates — that flag is set by the species labeler in stage 4
     and selects the best three angles per fish-grouping for
     length measurement.
  2. Of those, only images without a *completed* headtail label get
     a fresh LS task.

After importing tasks the activity also marks any pre-existing
incomplete headtail labels for the dive as `superseded=True`, so the
sync workflow's downstream consumers (calibration, measurement) can
ignore stale rows that were obsoleted by a re-import.
"""

from typing import List

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client

HEADTAIL_FOLDER = "headtail_jpeg"


def _select_target_images(
    species_labels: List[SpeciesLabel],
    images_by_id: dict[int, Image],
    existing_headtail_labels: List[HeadTailLabel],
) -> List[Image]:
    """Pick the images that need a fresh headtail LS task.

    Source: species labels marked `top_three_photos_of_group=True`.
    Filter: drop any image whose existing headtail label is already
    completed (so re-running is a no-op for finished work).
    """
    completed_ids = {
        label.image_id for label in existing_headtail_labels if label.completed
    }
    selected: List[Image] = []
    for label in species_labels:
        if not label.top_three_photos_of_group:
            continue
        if label.image_id in completed_ids:
            continue
        image = images_by_id.get(label.image_id)
        if image is not None:
            selected.append(image)
    return selected


def _build_task(image: Image) -> dict:
    return {
        "data": {"image": build_image_url(HEADTAIL_FOLDER, image.checksum)},
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
        species_labels = await fs.labels.get_species_labels(dive_id) or []
        existing_headtail = await fs.labels.get_headtail_labels(dive_id) or []

        # Hydrate Image rows by id for the species labels we care about.
        target_image_ids = {
            label.image_id
            for label in species_labels
            if label.top_three_photos_of_group
        }
        images_by_id: dict[int, Image] = {}
        for image_id in target_image_ids:
            image = await fs.images.get(image_id=image_id)
            if image is not None:
                images_by_id[image.id] = image

        targets = _select_target_images(
            species_labels, images_by_id, existing_headtail
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
                "Dive %d has no top-three species images needing headtail "
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

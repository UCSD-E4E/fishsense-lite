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

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client

# Physical Garage prefix the data-worker writes slate JPEGs to (stage 9).
# Was the nginx virtual name "dive_slate_jpgs"; now the real key prefix.
DIVE_SLATE_FOLDER = "preprocess_slate_images_jpeg"
SLATE_CONTENT_MARKER = "Slate, Laser on slate"


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


def _build_task(image: Image) -> dict:
    """Build an LS task. Emits both `image` and `img` keys to satisfy
    legacy LS labeling-config XML across prod projects — see
    `populate_laser_label_studio_project_activity._build_task`."""
    url = build_image_url(DIVE_SLATE_FOLDER, image.checksum)
    return {
        "data": {"image": url, "img": url},
        "predictions": [],
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

        new_count = 0
        images: List[Image] = []
        for image_id in target_ids:
            image = await fs.images.get(image_id=image_id)
            if image is not None:
                images.append(image)
            activity.heartbeat()

        if images:
            tasks = [_build_task(image) for image in images]

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

        # Supersede pass: retire previously-incomplete slate rows so new ones
        # are canonical (mirrors headtail/species). Only already-persisted rows.
        for old in existing_slate:
            if old.completed or old.superseded or old.id is None:
                continue
            old.superseded = True
            await fs.labels.put_dive_slate_label(old.image_id, old)
            activity.heartbeat()

        return new_count

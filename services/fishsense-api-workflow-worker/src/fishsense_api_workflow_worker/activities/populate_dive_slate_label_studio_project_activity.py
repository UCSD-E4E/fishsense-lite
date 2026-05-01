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

DIVE_SLATE_FOLDER = "dive_slate_jpgs"
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
    return {
        "data": {"image": build_image_url(DIVE_SLATE_FOLDER, image.checksum)},
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
        if not target_ids:
            activity.logger.info(
                "Dive %d has no slate-marked images needing labels; skipping",
                dive_id,
            )
            return 0

        images: List[Image] = []
        for image_id in target_ids:
            image = await fs.images.get(image_id=image_id)
            if image is not None:
                images.append(image)

        if not images:
            return 0

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
                label_studio_json={},
                user_id=None,
            )
            await fs.labels.put_dive_slate_label(image.id, label)

        return await import_tasks_and_record_labels(
            project_id=project_id,
            tasks=tasks,
            record_label=_record,
            items=images,
        )

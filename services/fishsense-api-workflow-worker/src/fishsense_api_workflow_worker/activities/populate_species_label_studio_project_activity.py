"""Activity to populate the species-labeling LS project for a dive.

Ports stage 4 of `populate_label_studio_project.ipynb`. Pushes one LS
task per image in the dive, then upserts a SpeciesLabel row that
anchors the (image, LS task, project) triple.

The notebook also pre-loaded predictions from `species_labels.csv` +
`species_map.csv` (a Google-Drive labeling pass migration) and from
SQL laser labels. That CSV bootstrap is intentionally dropped — it
was a one-shot import. The laser-keypoint prediction was also
dropped: emitting it requires the JPEG's pixel dimensions, which the
worker doesn't have at hand. Stage 4.2 sync still pulls SQL laser
data forward into SpeciesLabel rows post-labeling.
"""

from typing import List

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client

SPECIES_FOLDER = "groups_jpeg"


def _select_incomplete_images(
    images: List[Image], existing_labels: List[SpeciesLabel]
) -> List[Image]:
    """Return only images that need a fresh LS task — no species
    label yet, or one whose `completed` is not True."""
    by_image_id = {label.image_id: label for label in existing_labels}
    return [
        image
        for image in images
        if (label := by_image_id.get(image.id)) is None or not label.completed
    ]


def _build_task(image: Image) -> dict:
    return {
        "data": {"image": build_image_url(SPECIES_FOLDER, image.checksum)},
        "predictions": [],
        "annotations": [],
    }


@activity.defn
async def populate_species_label_studio_project_activity(
    dive_id: int, project_id: int
) -> int:
    """Push tasks for every incomplete image in `dive_id` to `project_id`.

    Returns the number of tasks imported (== rows upserted).
    """
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []
        existing_labels = await fs.labels.get_species_labels(dive_id) or []

        incomplete = _select_incomplete_images(images, existing_labels)
        if not incomplete:
            activity.logger.info(
                "Dive %d has no incomplete species images; nothing to import",
                dive_id,
            )
            return 0

        tasks = [_build_task(image) for image in incomplete]

        async def _record(image: Image, task_id: int) -> None:
            label = SpeciesLabel(
                id=None,
                image_id=image.id,
                label_studio_task_id=task_id,
                label_studio_project_id=project_id,
                image_url=build_image_url(SPECIES_FOLDER, image.checksum),
                updated_at=None,
                completed=False,
                label_studio_json={},
                user_id=None,
                grouping=None,
                top_three_photos_of_group=None,
                slate_upside_down=None,
                laser_x=None,
                laser_y=None,
                laser_label=None,
                content_of_image=None,
                fish_measurable_category=None,
                fish_angle_category=None,
                fish_curved_category=None,
            )
            await fs.labels.put_species_label(image.id, label)

        return await import_tasks_and_record_labels(
            project_id=project_id,
            tasks=tasks,
            record_label=_record,
            items=incomplete,
        )

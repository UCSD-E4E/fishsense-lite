"""Activity to populate the laser-labeling LS project for a dive.

Ports stage 0.3 of `populate_label_studio_project.ipynb` — pushes one
LS task per still-unlabeled image in the dive, then upserts a
LaserLabel row that anchors the (image, LS task, project) triple so
the existing sync workflow (`SyncLabelStudioLaserLabelsWorkflow`) can
pull annotated values back into SQL once a labeler completes them.
"""

from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client

PREPROCESS_FOLDER = "preprocess_jpeg"


def _select_unlabeled_images(
    images: List[Image], existing_labels: List[LaserLabel]
) -> List[Image]:
    """Return only images that need a fresh LS task — no completed
    LaserLabel exists for them in any project.

    Multi-row-aware: an image carrying both a completed row in
    project 43 and an incomplete sentinel row in project NULL is
    treated as labeled. The previous shape collapsed labels into a
    `{image_id: label}` dict and let SDK iteration order pick which
    row won, which silently leaked already-labeled images into the
    task-import set when the incomplete row happened to land last.
    """
    completed_image_ids = {
        label.image_id for label in existing_labels if label.completed
    }
    return [image for image in images if image.id not in completed_image_ids]


def _build_task(image: Image) -> dict:
    return {
        "data": {"image": build_image_url(PREPROCESS_FOLDER, image.checksum)},
        "annotations": [],
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
        existing_labels = await fs.labels.get_laser_labels(dive_id) or []

        unlabeled = _select_unlabeled_images(images, existing_labels)
        if not unlabeled:
            activity.logger.info(
                "Dive %d has a completed laser label for every image; "
                "nothing to import",
                dive_id,
            )
            return 0

        tasks = [_build_task(image) for image in unlabeled]

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

        return await import_tasks_and_record_labels(
            project_id=project_id,
            tasks=tasks,
            record_label=_record,
            items=unlabeled,
        )

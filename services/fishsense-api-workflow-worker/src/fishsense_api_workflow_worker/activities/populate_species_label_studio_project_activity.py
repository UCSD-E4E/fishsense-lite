"""Activity to populate the species-labeling LS project for a dive.

Cohort source flipped 2026-05-05 from "every image without a
completed species label" → "every image carrying a *valid* LaserLabel
without a non-sentinel species row." A laser label is "valid" when
`completed=True`, `superseded=False`, and both `x` and `y` are
populated — same gate `perform_laser_calibration_activity` and the
validator's `_positive_xy` already use as "usable laser." Cascading
from lasers lets species labeling kick off in parallel with head/tail
(stage 5.1) as soon as laser labelers + the validator sign off.

There is no supersede pass here — `SpeciesLabel` has no `superseded`
column (intentional, see CLAUDE.md `dive_pipeline_status` view). In
practice the cohort selector drops a dive the moment any image gets
a non-sentinel species row, so a re-fire on the same dive doesn't
re-import. Stale incomplete rows from a partial prior run will
linger visibly to downstream readers; an operator must drop them
manually if needed.
"""

from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client

# Physical Garage prefix the data-worker writes species JPEGs to
# (stage 2). Was the nginx virtual name "groups_jpeg".
SPECIES_FOLDER = "preprocess_groups_jpeg"


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
    existing_species_labels: List[SpeciesLabel],
) -> List[Image]:
    """Pick the images that need a fresh species LS task.

    Source: laser labels passing `_is_valid_laser`.
    Filter: drop any image whose existing species label is already
    completed (so re-running is a no-op for finished work).
    """
    completed_ids = {
        label.image_id for label in existing_species_labels if label.completed
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
    url = build_image_url(SPECIES_FOLDER, image.checksum)
    return {
        "data": {"image": url, "img": url},
        "predictions": [],
        "annotations": [],
    }


@activity.defn
async def populate_species_label_studio_project_activity(
    dive_id: int, project_id: int
) -> int:
    """Push species tasks for `dive_id` and supersede stale rows.

    Returns the number of tasks imported.
    """
    async with get_fs_client() as fs:
        laser_labels = await fs.labels.get_laser_labels(dive_id) or []
        existing_species = await fs.labels.get_species_labels(dive_id) or []

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
            laser_labels, images_by_id, existing_species
        )

        new_count = 0
        if targets:
            tasks = [_build_task(image) for image in targets]

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

            new_count = await import_tasks_and_record_labels(
                project_id=project_id,
                tasks=tasks,
                record_label=_record,
                items=targets,
            )
        else:
            activity.logger.info(
                "Dive %d has no laser-valid images needing species "
                "labels; skipping task import",
                dive_id,
            )

        return new_count

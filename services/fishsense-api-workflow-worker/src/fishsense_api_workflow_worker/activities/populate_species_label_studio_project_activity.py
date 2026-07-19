"""Activity to populate the species-labeling LS project for a dive.

Cohort source flipped 2026-05-05 from "every image without a
completed species label" → "every image carrying a *valid* LaserLabel
without a non-sentinel species row." A laser label is "valid" when
`completed=True`, `superseded=False`, and both `x` and `y` are
populated — same gate `perform_laser_calibration_activity` and the
validator's `_positive_xy` already use as "usable laser." Cascading
from lasers lets species labeling kick off in parallel with head/tail
(stage 5.1) as soon as laser labelers + the validator sign off.

The activity is **idempotent** so it can run on a schedule: an image
that already carries a non-superseded species row for the target
project is skipped (no duplicate LS task), and the end-of-run supersede
pass only dead-letters previously-incomplete rows belonging to a
*different* (stale/old) project — the target project's own in-progress
rows are left untouched. A re-run with no newly-laser-valid images
imports nothing and writes nothing. `SpeciesLabel` gained a
`superseded` column for uniform dead-letter semantics across all four
label types.
"""

from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_image_url,
    import_tasks_and_record_labels,
    publish_label_studio_project,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.object_store import open_object_store_client

# Physical Garage prefix the data-worker writes species JPEGs to
# (stage 2). Was the nginx virtual name "groups_jpeg".
SPECIES_FOLDER = "preprocess_groups_jpeg"


async def _gate_on_jpeg_presence(images: List[Image]) -> List[Image]:
    """Keep only images whose stage-2 species JPEG is already in Garage.

    When populate runs decoupled from preprocess (on its own schedule),
    seeding a species row for an image whose JPEG isn't written yet would
    satisfy the preprocess cohort's "has species row" exit — the image
    would then never get preprocessed and its LS task would point at a
    missing JPEG forever. Gating on JPEG existence defers those images to
    a later run (once preprocess has written them); it's a no-op when
    populate is chained right after preprocess (all JPEGs present).
    """
    if not images:
        return images
    store = open_object_store_client()
    present: List[Image] = []
    for image in images:
        if await store.has_processed_jpeg(SPECIES_FOLDER, image.checksum):
            present.append(image)
        else:
            activity.logger.info(
                "species JPEG not yet in Garage for image %d (checksum=%s); "
                "deferring to a later populate run",
                image.id,
                image.checksum,
            )
        activity.heartbeat()
    return present


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
    project_id: int,
) -> List[Image]:
    """Pick the images that need a fresh species LS task.

    Source: laser labels passing `_is_valid_laser`.
    Filter (idempotent — safe to schedule):
      - drop any image whose existing species label is already
        completed (finished work is a no-op on re-run);
      - drop any image that already carries a *non-superseded* species
        row for THIS project — a prior run already imported its task, so
        a scheduled re-run must not re-import a duplicate LS task.

    An image whose only species row is superseded, or belongs to a
    different (stale/old) project, is still selected: that's the
    migrate-onto-the-current-project path, and it stays idempotent
    because the row this run writes then satisfies the second filter on
    the next run.
    """
    completed_ids = {
        label.image_id for label in existing_species_labels if label.completed
    }
    already_in_project_ids = {
        label.image_id
        for label in existing_species_labels
        if label.label_studio_project_id == project_id and not label.superseded
    }
    selected: List[Image] = []
    for label in laser_labels:
        if not _is_valid_laser(label):
            continue
        if label.image_id in completed_ids:
            continue
        if label.image_id in already_in_project_ids:
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

        selected = _select_target_images(
            laser_labels, images_by_id, existing_species, project_id
        )
        # Only import tasks for images whose species JPEG is already in
        # Garage (see `_gate_on_jpeg_presence`). Deferred images stay in
        # the cohort and are picked up on a later run.
        targets = await _gate_on_jpeg_presence(selected)
        deferred = len(selected) - len(targets)

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
                    superseded=False,
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

        # Supersede pass: dead-letter previously-incomplete species rows that
        # belong to a *different* (stale/old) project so the current project's
        # rows are canonical. Rows already in `project_id` are this project's
        # live tasks and are left untouched — that's what makes a re-run
        # idempotent and the activity safe to schedule. Only acts on
        # already-persisted rows (id set); the fresh rows from _record above
        # aren't in `existing_species`.
        for old in existing_species:
            if old.completed or old.superseded or old.id is None:
                continue
            if old.label_studio_project_id == project_id:
                continue
            old.superseded = True
            await fs.labels.put_species_label(old.image_id, old)
            activity.heartbeat()

        # Publish only when the project's task set is COMPLETE. Species tasks
        # trickle in as stage-2 JPEGs are processed, so a run that deferred
        # any image (JPEG not yet in Garage) leaves the project a hidden
        # draft — never shown to annotators half-populated. Once nothing is
        # deferred, publish iff the project holds tasks (this run's imports or
        # a prior run's non-superseded rows for this project).
        already_in_project = any(
            label.label_studio_project_id == project_id and not label.superseded
            for label in existing_species
        )
        if deferred == 0 and (new_count > 0 or already_in_project):
            await publish_label_studio_project(project_id)

        return new_count

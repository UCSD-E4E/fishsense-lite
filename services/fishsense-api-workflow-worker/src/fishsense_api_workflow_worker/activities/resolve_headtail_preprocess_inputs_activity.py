"""Activity to resolve the per-image inputs stage 5.1 needs for a dive.

Returns a fully-populated `PreprocessHeadtailImagesInput` ready to
hand to the data-worker's child workflow. Image set is filtered to
images carrying a *valid* LaserLabel (completed=True, superseded=False,
both x/y populated) whose image has no non-sentinel HeadTailLabel
row. Matches the API selector predicate (cohort flipped from species
top-3 → valid lasers on 2026-05-04).
"""

from __future__ import annotations

from fishsense_shared import PreprocessHeadtailImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


def _is_valid_laser(label) -> bool:
    """Same predicate the API SQL uses for the cohort gate."""
    return bool(
        label.completed
        and not label.superseded
        and label.x is not None
        and label.y is not None
    )


@activity.defn
async def resolve_headtail_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessHeadtailImagesInput:
    activity.logger.info(
        "resolving headtail preprocess inputs dive_id=%d", dive_id
    )
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")

        intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if intrinsics is None:
            raise ValueError(
                f"camera_id={dive.camera_id} has no intrinsics"
            )

        laser_labels = await fs.labels.get_laser_labels(dive_id) or []
        existing_headtail = await fs.labels.get_headtail_labels(dive_id) or []
        # Skip sentinel rows (project_id IS NULL) — see API selector
        # docstring for rationale.
        labeled_ids = {
            label.image_id
            for label in existing_headtail
            if label.label_studio_project_id is not None
        }

        target_image_ids = [
            label.image_id
            for label in laser_labels
            if _is_valid_laser(label) and label.image_id not in labeled_ids
        ]

        images = await fs.images.get(dive_id=dive_id) or []
        checksum_by_id = {image.id: image.checksum for image in images}
        image_checksums = [
            checksum_by_id[image_id]
            for image_id in target_image_ids
            if image_id in checksum_by_id
        ]

        activity.logger.info(
            "resolved headtail preprocess inputs dive_id=%d "
            "valid_laser_targets=%d image_checksums=%d",
            dive_id,
            len(target_image_ids),
            len(image_checksums),
        )
        return PreprocessHeadtailImagesInput(
            dive_id=dive_id,
            image_checksums=image_checksums,
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )

"""Activity to resolve the per-image inputs stage 5.1 needs for a dive.

Returns a fully-populated `PreprocessHeadtailImagesInput` ready to
hand to the data-worker's child workflow. Image set is filtered to
species labels with `top_three_photos_of_group=True` whose image has
no HeadTailLabel row at all — once populate seeds a (possibly
incomplete) row, the headtail JPEG is on the file-exchange and we
don't regenerate it. Matches the API selector predicate.
"""

from __future__ import annotations

from fishsense_shared import PreprocessHeadtailImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def resolve_headtail_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessHeadtailImagesInput:
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

        species_labels = await fs.labels.get_species_labels(dive_id) or []
        existing_headtail = await fs.labels.get_headtail_labels(dive_id) or []
        labeled_ids = {label.image_id for label in existing_headtail}

        target_image_ids = [
            label.image_id
            for label in species_labels
            if label.top_three_photos_of_group
            and label.image_id not in labeled_ids
        ]

        images = await fs.images.get(dive_id=dive_id) or []
        checksum_by_id = {image.id: image.checksum for image in images}
        image_checksums = [
            checksum_by_id[image_id]
            for image_id in target_image_ids
            if image_id in checksum_by_id
        ]

        return PreprocessHeadtailImagesInput(
            dive_id=dive_id,
            image_checksums=image_checksums,
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )

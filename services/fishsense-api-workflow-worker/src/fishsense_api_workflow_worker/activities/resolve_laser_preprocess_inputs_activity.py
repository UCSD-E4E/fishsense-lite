"""Activity to resolve the per-image inputs stage 0.1 needs for a dive.

Returns a fully-populated `PreprocessLaserImagesInput` ready to hand
to the data-worker's child workflow. Image checksum filter matches
stage 0.3's populate predicate (laser label missing or
`completed=False`), so a re-run after some labels complete naturally
narrows the work.

Default bbox is the original notebook constant — kept here rather
than baked into the data-worker so the api-worker can swap to a
per-camera bbox once we ship a second sensor without touching the
data-worker.
"""

from __future__ import annotations

from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_shared import PreprocessLaserImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

DEFAULT_LASER_BBOX: List[int] = [1800, 700, 2400, 1600]


def _select_incomplete_images(
    images: List[Image], existing_labels: List[LaserLabel]
) -> List[Image]:
    by_image_id = {label.image_id: label for label in existing_labels}
    return [
        image
        for image in images
        if (label := by_image_id.get(image.id)) is None or not label.completed
    ]


@activity.defn
async def resolve_laser_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessLaserImagesInput:
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

        images = await fs.images.get(dive_id=dive_id) or []
        existing_labels = await fs.labels.get_laser_labels(dive_id) or []
        incomplete = _select_incomplete_images(images, existing_labels)

        return PreprocessLaserImagesInput(
            dive_id=dive_id,
            image_checksums=[image.checksum for image in incomplete],
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
            bbox=list(DEFAULT_LASER_BBOX),
        )

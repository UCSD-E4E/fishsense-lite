"""Activity to resolve the per-image inputs stage 0.1 needs for a dive.

Returns the image checksums whose laser label is missing or not yet
completed (the same `incomplete` predicate stage 0.3 populate uses,
see `populate_laser_label_studio_project_activity`), plus the dive's
camera intrinsics flattened to plain lists for activity-payload
serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client


@dataclass
class LaserPreprocessInputs:
    """Resolved inputs for one dive's stage 0.1 fan-out."""

    dive_id: int
    image_checksums: List[str]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


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
) -> LaserPreprocessInputs:
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

        return LaserPreprocessInputs(
            dive_id=dive_id,
            image_checksums=[image.checksum for image in incomplete],
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )

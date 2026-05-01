"""Activity to resolve the per-cluster inputs stage 2 needs for a dive.

Returns a fully-populated `PreprocessDiveImagesInput` ready to hand
to the data-worker's child workflow. Clusters preserve the temporal
grouping from `DiveFrameCluster(data_source=PREDICTION)` so the
per-image overlay can render "image i of N" for each cluster.
"""

from __future__ import annotations

from typing import List

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_shared import PreprocessDiveImagesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def resolve_dive_image_preprocess_inputs_activity(
    dive_id: int,
) -> PreprocessDiveImagesInput:
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

        prediction_clusters = (
            await fs.images.get_clusters(dive_id, DataSource.PREDICTION.value)
            or []
        )
        images = await fs.images.get(dive_id=dive_id) or []
        checksum_by_id = {image.id: image.checksum for image in images}

        # Drop image_ids that no longer have an image row (defensive — a
        # cluster pointing at a deleted image would fail downstream
        # download_raw with a confusing 404).
        clusters: List[List[str]] = []
        for cluster in prediction_clusters:
            cluster_checksums = [
                checksum_by_id[image_id]
                for image_id in (cluster.image_ids or [])
                if image_id in checksum_by_id
            ]
            if cluster_checksums:
                clusters.append(cluster_checksums)

        return PreprocessDiveImagesInput(
            dive_id=dive_id,
            clusters=clusters,
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
        )

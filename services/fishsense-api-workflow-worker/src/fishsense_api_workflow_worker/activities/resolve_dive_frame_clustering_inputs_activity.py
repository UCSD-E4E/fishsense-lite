"""Activity to resolve the per-image inputs stage 1 needs for a dive.

Returns a fully-populated `ClusterDiveFramesInput` ready to hand to
the data-worker's child workflow. The kernel only needs
`(image_id, taken_datetime)` pairs — image bytes are never read in
stage 1, so this resolver does no NAS or file-exchange staging.
"""

from __future__ import annotations

from fishsense_shared import ClusterDiveFrameImage, ClusterDiveFramesInput
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def resolve_dive_frame_clustering_inputs_activity(
    dive_id: int,
) -> ClusterDiveFramesInput:
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []

    return ClusterDiveFramesInput(
        dive_id=dive_id,
        images=[
            ClusterDiveFrameImage(
                image_id=image.id,
                taken_datetime=image.taken_datetime,
            )
            for image in images
            if image.id is not None
        ],
    )

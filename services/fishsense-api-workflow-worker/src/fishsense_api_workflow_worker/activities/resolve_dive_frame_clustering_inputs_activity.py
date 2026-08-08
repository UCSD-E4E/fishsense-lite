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
    activity.logger.info(
        "resolving clustering inputs dive_id=%d", dive_id
    )
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []
        # Canonical frames only. The same physical frames live under several
        # dive rows (half of prod's image table is duplicate content), and
        # `is_canonical` marks which copy is the real one. The cohort selectors
        # gate on it, and CLAUDE.md requires resolvers to mirror the selector
        # predicate exactly -- otherwise the dispatched per-image work would not
        # match what the cohort promised, and the dive could never drain.
        # This also covers on-demand/backfill runs, which bypass the cohort.
        images = [image for image in images if image.is_canonical]
    cluster_images = [
        ClusterDiveFrameImage(
            image_id=image.id,
            taken_datetime=image.taken_datetime,
        )
        for image in images
        if image.id is not None
    ]
    activity.logger.info(
        "resolved clustering inputs dive_id=%d images=%d cluster_inputs=%d",
        dive_id,
        len(images),
        len(cluster_images),
    )
    return ClusterDiveFramesInput(dive_id=dive_id, images=cluster_images)

"""Activity to persist stage-1 PREDICTION clusters for a dive.

Takes the `list[list[int]]` output from the data-worker child
(`DiveFrameClusteringWorkflow`) and POSTs one
`DiveFrameCluster(data_source=PREDICTION)` per cluster via the SDK.

Idempotency note: stage-1 clustering is one-shot per dive (the cohort
selector excludes dives that already have any PREDICTION cluster), so
this activity should not be re-run against a dive that already has
PREDICTION clusters. If a previous parent run posted partial clusters
and then failed, the cohort selector will continue to skip the dive
on re-fires (the "no PREDICTION cluster" gate is met by even one
existing cluster). Recovery: an operator must drop the partial
PREDICTION cluster rows manually before re-firing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def persist_dive_frame_clusters_activity(
    dive_id: int, clusters: List[List[int]]
) -> int:
    """POST one PREDICTION DiveFrameCluster per id-list in `clusters`.

    Returns the number of clusters posted.
    """
    if not clusters:
        activity.logger.info(
            "no PREDICTION clusters to post for dive_id=%d (empty cluster list)",
            dive_id,
        )
        return 0

    posted = 0
    now = datetime.now(timezone.utc)
    async with get_fs_client() as fs:
        for image_ids in clusters:
            if not image_ids:
                continue
            cluster = DiveFrameCluster(
                id=None,
                dive_id=dive_id,
                image_ids=image_ids,
                data_source=DataSource.PREDICTION,
                fish_id=None,
                updated_at=now,
            )
            await fs.images.post_cluster(dive_id, cluster)
            posted += 1
            activity.heartbeat()

    activity.logger.info(
        "posted %d PREDICTION clusters for dive_id=%d", posted, dive_id
    )
    return posted

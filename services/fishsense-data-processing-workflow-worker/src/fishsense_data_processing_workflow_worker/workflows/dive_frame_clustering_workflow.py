"""Stage 1 (dive-frame clustering) workflow on the data-worker.

Inputs are pre-resolved by the api-worker parent
(`ClusterDiveFramesParentWorkflow` on `fishsense_api_queue`), which
fetches dive images via the SDK and packs `(image_id, taken_datetime)`
pairs into a `ClusterDiveFramesInput`. This workflow only delegates
to the cluster activity — no SDK or NAS access on the data-worker
side. Output is `list[list[int]]` (image_ids per cluster); the parent
persists each list via `images.post_cluster(data_source=PREDICTION)`.
"""

from datetime import timedelta
from typing import List

from fishsense_shared import ClusterDiveFramesInput
from temporalio import workflow


@workflow.defn
class DiveFrameClusteringWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: ClusterDiveFramesInput) -> List[List[int]]:
        workflow.logger.info(
            "clustering dive_id=%d images=%d",
            payload.dive_id,
            len(payload.images),
        )

        return await workflow.execute_activity(
            "cluster_dive_frames",
            payload.images,
            schedule_to_close_timeout=timedelta(minutes=10),
        )

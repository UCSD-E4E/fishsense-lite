"""Stage 1 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing dive-frame clustering and
dispatches `DiveFrameClusteringWorkflow` to the data-worker. After
the child returns, persists each cluster as a PREDICTION
DiveFrameCluster row via the SDK so stage-2 species preprocessing has
the cluster gate it depends on.

Cluster-correctness invariants mirror the four preprocess parents:
  * `ScheduleOverlapPolicy.SKIP` on the schedule prevents racing
    selectors picking the same dive concurrently.
  * Child workflow id is deterministic (`cluster-{dive_id}`) and
    dispatched with `ALLOW_DUPLICATE_FAILED_ONLY`. If a parent run
    races past the schedule guard (e.g. manual + scheduled overlap),
    the second `start_child_workflow` raises
    `WorkflowAlreadyStartedError` which the parent catches; persist
    then runs against the already-completed child's output via a
    fresh activity call.

Stage 1 has no NAS or file-exchange staging — clustering is pure
math on `(image_id, taken_datetime)` pairs, no image bytes needed.
That keeps the parent's activity sequence to selector → resolver →
child → persist.
"""

from datetime import timedelta
from typing import List

from fishsense_shared import ClusterDiveFramesInput
from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SDK_FAIL_FAST_RETRY_POLICY,
    )

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class ClusterDiveFramesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing dive-frame
    clustering and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_clustering_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_dive_frame_clustering_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            result_type=ClusterDiveFramesInput,
        )

        workflow.logger.info(
            "dispatching dive-frame clustering to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.images),
        )

        if not inputs.images:
            return inputs.dive_id

        clusters: List[List[int]] = []
        try:
            clusters = await workflow.execute_child_workflow(
                "DiveFrameClusteringWorkflow",
                inputs,
                id=f"cluster-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(minutes=15),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
                result_type=list,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "cluster-%d already ran successfully in a prior firing; "
                "skipping data-worker dispatch — persist will no-op via "
                "the cohort selector on the next firing",
                dive_id,
            )
            return inputs.dive_id

        await workflow.execute_activity(
            "persist_dive_frame_clusters_activity",
            args=(dive_id, clusters),
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=2),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )

        return inputs.dive_id

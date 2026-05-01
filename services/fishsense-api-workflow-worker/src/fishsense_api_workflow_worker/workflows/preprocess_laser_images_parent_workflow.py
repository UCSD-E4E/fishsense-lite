"""Stage 0.1 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing laser preprocessing, resolves
its incomplete-image-set + camera intrinsics via SDK, and dispatches
the resolved inputs to the data-worker's `PreprocessLaserImagesWorkflow`
on `fishsense_data_processing_queue`.

Cluster-correctness invariants — relevant once the data-worker scales
beyond a single replica:

* Per-image activities (in the child workflow) PUT to nginx DAV which
  is idempotent under overwrite. Retried activities don't double-write.
* SDK upserts on the resolver side don't mutate state — read-only.
* The schedule that fires this workflow uses
  `overlap_policy=ScheduleOverlapPolicy.SKIP`, so a run still in
  flight when the next firing arrives is dropped at the schedule level.
* The child workflow is started with a deterministic id
  (`preprocess-laser-{dive_id}`); if a parent run somehow races past
  the schedule guard (manual + scheduled trigger overlap), the second
  child-workflow start hits `WorkflowAlreadyStarted` and is a no-op
  rather than redoing the dive.
"""

from datetime import timedelta

from fishsense_shared import PreprocessLaserImagesInput
from temporalio import workflow

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class PreprocessLaserImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive without laser extrinsics
    and dispatch its preprocessing to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    Each invocation drains exactly one dive — an N-dive backlog clears
    in N hourly schedule firings.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_laser_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_laser_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            result_type=PreprocessLaserImagesInput,
        )

        workflow.logger.info(
            "dispatching laser preprocess to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        await workflow.execute_child_workflow(
            "PreprocessLaserImagesWorkflow",
            inputs,
            id=f"preprocess-laser-{dive_id}",
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            execution_timeout=timedelta(hours=1),
        )

        return inputs.dive_id

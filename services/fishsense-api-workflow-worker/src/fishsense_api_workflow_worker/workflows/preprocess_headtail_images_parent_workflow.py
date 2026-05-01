"""Stage 5.1 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing head/tail preprocessing and
dispatches `PreprocessHeadtailImagesWorkflow` to the data-worker.

Same cluster-correctness invariants as
`PreprocessLaserImagesParentWorkflow` — see CLAUDE.md.
"""

from datetime import timedelta

from fishsense_shared import PreprocessHeadtailImagesInput
from temporalio import workflow

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class PreprocessHeadtailImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing head/tail
    preprocessing and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_headtail_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_headtail_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            result_type=PreprocessHeadtailImagesInput,
        )

        workflow.logger.info(
            "dispatching headtail preprocess to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        await workflow.execute_child_workflow(
            "PreprocessHeadtailImagesWorkflow",
            inputs,
            id=f"preprocess-headtail-{dive_id}",
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            execution_timeout=timedelta(hours=1),
        )

        return inputs.dive_id

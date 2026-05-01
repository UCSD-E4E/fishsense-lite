"""Stage 9 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing slate preprocessing and
dispatches `PreprocessSlateImagesWorkflow` to the data-worker.

Same cluster-correctness invariants as
`PreprocessLaserImagesParentWorkflow` — see CLAUDE.md.
"""

from datetime import timedelta

from fishsense_shared import PreprocessSlateImagesInput
from temporalio import workflow

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class PreprocessSlateImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing slate preprocessing
    and dispatch its work to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty).
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_slate_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_slate_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            result_type=PreprocessSlateImagesInput,
        )

        workflow.logger.info(
            "dispatching slate preprocess to data-worker dive_id=%d images=%d slate_id=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
            inputs.slate_id,
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        # Slate stage needs both the raw .ORFs and the slate template
        # PDF on the file-exchange before the data-worker child runs.
        await workflow.execute_activity(
            "stage_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
        )
        await workflow.execute_activity(
            "stage_slate_pdf_activity",
            args=(inputs.slate_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

        await workflow.execute_child_workflow(
            "PreprocessSlateImagesWorkflow",
            inputs,
            id=f"preprocess-slate-{dive_id}",
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            execution_timeout=timedelta(hours=1),
        )

        return inputs.dive_id

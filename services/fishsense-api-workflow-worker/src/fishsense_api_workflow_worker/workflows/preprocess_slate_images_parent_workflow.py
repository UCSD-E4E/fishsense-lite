"""Stage 9 parent workflow (api-worker side).

Picks the next HIGH-priority dive needing slate preprocessing and
dispatches `PreprocessSlateImagesWorkflow` to the data-worker. After
archive+cleanup, chains into
`PopulateDiveSlateLabelStudioProjectWorkflow` so slate JPEGs land in
the dive-slate LS project in the same hourly firing that produced
them.

Same cluster-correctness invariants as
`PreprocessLaserImagesParentWorkflow` — see CLAUDE.md. Populate child
uses deterministic id `populate-dive-slate-{dive_id}` so re-firings
on the same cohort dive no-op via WorkflowAlreadyStarted.
"""

from datetime import timedelta

from fishsense_shared import PreprocessSlateImagesInput
from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SCALING_RETRY_POLICY,
        SDK_FAIL_FAST_RETRY_POLICY,
    )

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
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_slate_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
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

        # Wake the NRP data-worker before its child workflow lands on
        # the queue (it scales to zero when idle). Idempotent — converges
        # on the configured replica count, never accumulates; a no-op
        # when k8s scaling isn't configured. Returns immediately, so the
        # pod's cold start overlaps the NAS-staging steps below.
        await workflow.execute_activity(
            "ensure_data_worker_running_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

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

        try:
            await workflow.execute_child_workflow(
                "PreprocessSlateImagesWorkflow",
                inputs,
                id=f"preprocess-slate-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=1),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "preprocess-slate-%d already ran successfully in a prior "
                "firing; skipping data-worker dispatch and continuing to "
                "cleanup + populate",
                dive_id,
            )

        # Drop the staged raw `.ORF` scratch objects from Garage; the
        # JPEGs stay (LS reads them via presign). NAS is never touched.
        await workflow.execute_activity(
            "cleanup_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=5),
        )

        try:
            await workflow.execute_child_workflow(
                "PopulateDiveSlateLabelStudioProjectWorkflow",
                dive_id,
                id=f"populate-dive-slate-{dive_id}",
                execution_timeout=timedelta(minutes=30),
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "populate-dive-slate-%d already ran in a prior hourly firing; "
                "skipping LS task import",
                dive_id,
            )

        return inputs.dive_id

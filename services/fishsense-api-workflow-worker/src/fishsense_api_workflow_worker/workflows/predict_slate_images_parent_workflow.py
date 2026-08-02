"""Slate-detector parent workflow (api-worker side).

Model-assisted slate labeling. Picks the next HIGH-priority dive needing slate
predictions, resolves its unpredicted slate-frame set + template + camera
intrinsics via SDK, stages the raw `.ORF` bytes AND the slate template PDF, and
dispatches the data-worker's CPU `PredictSlateImagesWorkflow`. The child returns
one gated prediction per frame; the parent persists them
(`put_slate_prediction`) so the dive-slate populate step can serve them as Label
Studio pre-annotations (assisted review).

Cohort: HIGH-priority + `dive_slate_id` set + at least one slate frame with no
`SlatePrediction` and no completed `DiveSlateLabel` (see
`select_next_for_slate_prediction`). One-shot per frame — a dive drops out once
every slate frame is predicted.

Same cluster-correctness invariants as the laser-predict parent: `overlap=SKIP`;
deterministic child id (`predict-slate-{dive_id}`) with `ALLOW_DUPLICATE` so a
dive can re-predict frames that became eligible later; per-image predict is
read-only against Garage + the SDK upsert is idempotent.
"""

from datetime import timedelta
from typing import List

from fishsense_shared import PredictSlateImagesInput, SlatePredictionResult
from temporalio import workflow
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SCALING_RETRY_POLICY,
        SDK_FAIL_FAST_RETRY_POLICY,
        STAGE_RAW_RETRY_POLICY,
    )

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"


@workflow.defn
class PredictSlateImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing slate predictions and
    dispatch the CPU slate detector to the data-worker. Returns the dive_id
    processed (or None when the backlog is empty) — each invocation drains
    exactly one dive.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_slate_prediction_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_slate_predict_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            result_type=PredictSlateImagesInput,
        )

        workflow.logger.info(
            "dispatching slate predict to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.images),
        )

        if not inputs.images:
            return inputs.dive_id

        # Wake the NRP data-worker before its child lands on the queue (scales
        # to zero when idle). Idempotent; no-op when k8s scaling isn't set.
        await workflow.execute_activity(
            "ensure_data_worker_running_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

        # Stage the raw `.ORF` frames AND the slate template PDF — the predict
        # activity rectifies the raw frame and renders the PDF into the template.
        await workflow.execute_activity(
            "stage_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=STAGE_RAW_RETRY_POLICY,
        )
        await workflow.execute_activity(
            "stage_slate_pdf_activity",
            args=(inputs.slate_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=STAGE_RAW_RETRY_POLICY,
        )

        results: List[SlatePredictionResult] = []
        try:
            results = await workflow.execute_child_workflow(
                "PredictSlateImagesWorkflow",
                inputs,
                id=f"predict-slate-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=2),
                # ALLOW_DUPLICATE so a dive can re-predict frames that became
                # eligible after a prior run; the resolver returns only
                # still-unpredicted frames and put_slate_prediction upserts.
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                result_type=List[SlatePredictionResult],
            )
        except WorkflowAlreadyStartedError:
            workflow.logger.info(
                "predict-slate-%d already running; skipping duplicate dispatch",
                dive_id,
            )

        if results:
            await workflow.execute_activity(
                "persist_slate_predictions_activity",
                args=(results,),
                schedule_to_close_timeout=timedelta(minutes=15),
                retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            )
            # Attach the freshly-persisted predictions to any *existing* dive-
            # slate LS tasks. The populate seeds pre-annotations only at import
            # time and runs once per dive, so a dive already populated before it
            # was predicted would otherwise never surface them. Idempotent
            # (skips tasks that already carry a slate-detector prediction).
            await workflow.execute_activity(
                "backfill_slate_predictions_for_dive_activity",
                args=(dive_id,),
                schedule_to_close_timeout=timedelta(minutes=15),
                retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            )

        # Drop the staged raw `.ORF` scratch from Garage; the NAS source is
        # never touched.
        await workflow.execute_activity(
            "cleanup_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=5),
        )

        return inputs.dive_id

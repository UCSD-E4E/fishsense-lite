"""Laser-detector parent workflow (api-worker side).

Model-assisted laser labeling. Picks the next HIGH-priority dive needing
laser predictions, resolves its unpredicted-image set + camera intrinsics via
SDK, stages the raw `.ORF` bytes, and dispatches the data-worker's GPU
`PredictLaserImagesWorkflow`. The child returns one prediction per image; the
parent persists them (`put_laser_prediction`) so the laser populate step can
serve them as Label Studio pre-annotations (assisted review).

Cohort: HIGH-priority + at least one image with no `LaserPrediction` and no
non-sentinel `LaserLabel` (see `select_next_for_laser_prediction`). One-shot
per image — a dive drops out once every image is predicted.

Same cluster-correctness invariants as the preprocess parents: the schedule
fires with `overlap=SKIP`; the child id is deterministic
(`predict-laser-{dive_id}`) with `ALLOW_DUPLICATE` so a dive can re-predict
images that became eligible later; per-image predict activities are read-only
against Garage + the SDK upsert is idempotent.
"""

from datetime import timedelta
from typing import List

from fishsense_shared import LaserPredictionResult, PredictLaserImagesInput
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
class PredictLaserImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing laser predictions and
    dispatch the GPU detector to the data-worker. Returns the dive_id
    processed (or None when the backlog is empty) — each invocation drains
    exactly one dive.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_laser_prediction_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
        )
        if dive_id is None:
            return None

        inputs = await workflow.execute_activity(
            "resolve_laser_predict_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SDK_FAIL_FAST_RETRY_POLICY,
            result_type=PredictLaserImagesInput,
        )

        workflow.logger.info(
            "dispatching laser predict to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.images),
        )

        if not inputs.images:
            return inputs.dive_id

        # Wake the NRP GPU data-worker before its child lands on the queue
        # (scales to zero when idle). Idempotent; no-op when k8s scaling
        # isn't configured.
        await workflow.execute_activity(
            "ensure_data_worker_running_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

        await workflow.execute_activity(
            "stage_raw_bytes_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=STAGE_RAW_RETRY_POLICY,
        )

        results: List[LaserPredictionResult] = []
        try:
            results = await workflow.execute_child_workflow(
                "PredictLaserImagesWorkflow",
                inputs,
                id=f"predict-laser-{dive_id}",
                task_queue=DATA_PROCESSING_TASK_QUEUE,
                execution_timeout=timedelta(hours=2),
                # ALLOW_DUPLICATE so a dive can re-predict images that became
                # eligible after a prior run; the resolver returns only
                # still-unpredicted images and put_laser_prediction upserts.
                id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
                result_type=List[LaserPredictionResult],
            )
        except WorkflowAlreadyStartedError:
            # A prior child with this id is still running (manual run
            # overlapping the schedule). It's doing the work; skip persist
            # this firing and let cleanup run.
            workflow.logger.info(
                "predict-laser-%d already running; skipping duplicate dispatch",
                dive_id,
            )

        if results:
            await workflow.execute_activity(
                "persist_laser_predictions_activity",
                args=(results,),
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

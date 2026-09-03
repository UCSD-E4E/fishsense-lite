"""Head/tail-detector parent workflow (api-worker side).

Model-assisted head/tail labeling. Picks the next HIGH-priority dive needing
predictions, resolves its still-unpredicted images and their laser dots via the
SDK, and dispatches the data-worker's GPU `PredictHeadtailImagesWorkflow`. The
child returns one result per image; the parent persists them so the head/tail
populate step can serve them as Label Studio pre-annotations.

**Lighter than the other predict parents, and deliberately so.** There is no
`stage_raw` and no `cleanup_raw`: the stage reads the stage-5.1 JPEG that is
already in Garage, which is the exact frame the labeler is shown. That removes
the NAS entirely from this path, and with it the failure mode that leaked 1,094
staged objects when the laser backfill first ran — there is nothing here to
leak.

Cohort: HIGH-priority + at least one canonical image with a *valid* laser dot,
no live human `HeadTailLabel`, and either no prediction or a stale one (see
`select_next_for_headtail_prediction`). Stale means a `predictor_version`
mismatch or a prediction made from a laser since superseded, so improving the
stage or cleaning up lasers drains as a cohort rather than needing a backfill.

Dispatches to the **GPU queue**, like the laser and slate predict parents. When
`wake_gpu_worker` reports that neither the GPU Deployment nor the CPU fallback
could start, this returns before dispatching: a child sent to an unserved queue
does not fail, it sits Running until its execution timeout. The dive stays in
the cohort for the next firing.
"""

from datetime import timedelta
from typing import List

from fishsense_shared.preprocess_contracts import (
    HeadtailPredictionResult,
    PredictHeadtailImagesInput,
)
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.activities.gpu_fallback import MODE_UNAVAILABLE

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class PredictHeadtailImagesParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing head/tail predictions and
    dispatch the GPU detector. Returns the dive_id processed (or None when the
    backlog is empty) — each invocation drains exactly one dive.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_headtail_prediction_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_headtail_predict_inputs_activity",
            dive_id,
            PredictHeadtailImagesInput,
        )

        workflow.logger.info(
            "dispatching headtail predict to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.images),
        )

        if not inputs.images:
            return inputs.dive_id

        mode = await _dispatch.wake_gpu_worker()
        if mode == MODE_UNAVAILABLE:
            workflow.logger.warning(
                "no worker available for the headtail-predict queue; "
                "skipping dive_id=%d this firing",
                inputs.dive_id,
            )
            return None

        workflow.logger.info("headtail predict running on %s capacity", mode)
        results: List[HeadtailPredictionResult] = await _dispatch.dispatch_child(
            "PredictHeadtailImagesWorkflow",
            inputs,
            child_id=f"predict-headtail-{dive_id}",
            # Generous enough to cover the CPU fallback, which runs the same
            # weights without a GPU and is far slower per image.
            execution_timeout=timedelta(hours=6),
            result_type=List[HeadtailPredictionResult],
            task_queue=_dispatch.DATA_PROCESSING_GPU_TASK_QUEUE,
        )

        if results:
            await _dispatch.run_sdk_activity(
                "persist_headtail_predictions_activity", results
            )

        return inputs.dive_id

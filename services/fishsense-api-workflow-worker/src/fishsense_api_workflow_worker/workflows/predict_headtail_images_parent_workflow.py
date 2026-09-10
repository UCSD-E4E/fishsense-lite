"""Head/tail-detector parent workflow (api-worker side).

Model-assisted head/tail labeling. Picks the next HIGH-priority dive needing
predictions, resolves its still-unpredicted images and their laser dots via the
SDK, and dispatches the data-worker's GPU `PredictHeadtailImagesWorkflow`. The
child returns one result per image; the parent persists them so the head/tail
populate step can serve them as Label Studio pre-annotations.

**The backfill call is unconditional (2026-09-10), and that changed this
workflow's command sequence.** It used to be gated on there being something to
persist. A workflow's command order is its replay contract, so this was
deployed against a drained queue -- schedule paused, no run in flight -- rather
than folded into a rolling deploy. If the gate is ever reintroduced or moved,
do the same; a run mid-flight across the change replays into a
non-determinism error.

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
    HEADTAIL_STATUS_NO_UPGRADE_AVAILABLE,
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

        # Drop the images this worker could not improve on. A GPU-less worker
        # returns these for rows that are already fallback-tier, because
        # rewriting them would replace a row with an identical one -- and a
        # fallback row is permanently stale by design, so the cohort re-offers
        # it every hour until a GPU can actually upgrade it. Persisting the
        # skip marker instead would blank a perfectly good prediction.
        persistable = [
            r for r in results if r.status != HEADTAIL_STATUS_NO_UPGRADE_AVAILABLE
        ]

        if persistable:
            await _dispatch.run_sdk_activity(
                "persist_headtail_predictions_activity", persistable
            )

        # Populate seeds a task's pre-annotation once, at import time, and
        # dedupes by URL — so for a dive whose tasks already exist, persisting
        # alone changes the database and nothing the labeler sees. That is most
        # of the corpus: 3,147 still-unlabelled tasks across 19 dives were
        # imported long before any prediction existed. Attaching to the
        # existing tasks is what makes a prediction visible. Last on purpose:
        # it is the only step whose failure leaves nothing to clean up, and
        # `BackfillHeadtailPredictionsWorkflow` can repair a dive afterwards on
        # its own.
        #
        # **Unconditional**, and that is the point rather than an oversight.
        # It also points the LS project's `model_version` at the tier the dive
        # holds, without which every attached prediction is invisible to the
        # labeler. Gated on `persistable` it could never run for the dive that
        # needs it most: an all-fallback dive re-offered on CPU capacity
        # returns nothing but `NO_UPGRADE_AVAILABLE`, so the list is empty
        # while its project may still be showing nothing at all. The activity
        # is idempotent and returns early when a dive has no attachable task,
        # so the cost of running it every firing is one cheap call.
        await _dispatch.run_sdk_activity(
            "backfill_headtail_predictions_for_dive_activity", dive_id
        )

        return inputs.dive_id

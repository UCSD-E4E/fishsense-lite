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

**This is the only parent that dispatches to the GPU queue.** The data-worker
is split in two — `fishsense_data_processing_queue` for the CPU stages,
`fishsense_data_processing_gpu_queue` for the laser detector — so nothing else
in the pipeline is gated on a GPU being schedulable. The GPU queue is served by
either the GPU Deployment or, after repeated failed starts, a CPU-only one
running the same checkpoint slowly; `wake_gpu_worker` decides and reports which
(see `activities.gpu_fallback`). When it reports that neither could start, this
parent returns **before staging any bytes**: the dive is left in the cohort and
picked up next hour, which is strictly better than staging its raw `.ORF`s from
the NAS and then hanging a child on an unserved queue for two hours.
"""

from datetime import timedelta
from typing import List

from fishsense_shared import (
    LaserAutoAcceptSummary,
    LaserPredictionResult,
    PredictLaserImagesInput,
)
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.activities.gpu_fallback import MODE_UNAVAILABLE

from fishsense_api_workflow_worker.workflows import _dispatch


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
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_laser_prediction_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_laser_predict_inputs_activity",
            dive_id,
            PredictLaserImagesInput,
        )

        workflow.logger.info(
            "dispatching laser predict to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.images),
        )

        if not inputs.images:
            return inputs.dive_id

        mode = await _dispatch.wake_gpu_worker()
        if mode == MODE_UNAVAILABLE:
            # Nothing is serving the GPU queue — not the GPU worker, not the
            # CPU fallback. Bail before staging: a child dispatched now would
            # not fail, it would sit Running until its two-hour execution
            # timeout. The dive stays in the cohort for the next firing.
            workflow.logger.warning(
                "no worker available for the laser-predict queue; "
                "skipping dive_id=%d this firing",
                inputs.dive_id,
            )
            return None

        workflow.logger.info("laser predict running on %s capacity", mode)
        await _dispatch.stage_raw(dive_id)
        results: List[LaserPredictionResult] = await _dispatch.dispatch_child(
            "PredictLaserImagesWorkflow",
            inputs,
            child_id=f"predict-laser-{dive_id}",
            # Generous enough to cover the CPU fallback, which runs the same
            # checkpoint without a GPU and is far slower per image.
            execution_timeout=timedelta(hours=6),
            result_type=List[LaserPredictionResult],
            task_queue=_dispatch.DATA_PROCESSING_GPU_TASK_QUEUE,
        )

        if results:
            await _dispatch.run_sdk_activity(
                "persist_laser_predictions_activity", results
            )

        # Before the backfill, deliberately. The scratch exists for the child,
        # which is done; holding it until a *Label Studio* step succeeds means
        # an LS outage leaks a dive's raw `.ORF`s into Garage. That is not
        # hypothetical — the first production run of the backfill failed on an
        # unregistered activity and left 1,094 staged objects behind.
        await _dispatch.cleanup_raw(dive_id)

        if results:
            # Populate seeds a task's pre-annotation once, at import time, and
            # dedupes by URL — so for a dive whose tasks already exist (which
            # is every dive the re-prediction cohort selects, since it only
            # picks dives still being labeled) persisting alone changes the
            # database and nothing the labeler sees. Attaching to the existing
            # tasks is what makes a new prediction visible.
            #
            # Last on purpose: it is the only step whose failure leaves nothing
            # to clean up, and `BackfillLaserPredictionsWorkflow` can repair a
            # dive on its own afterwards.
            await _dispatch.run_sdk_activity(
                "backfill_laser_predictions_for_dive_activity", dive_id
            )

        if results:
            # Judge the dive's predictions against its own laser line and
            # record which may skip human review. Appended at the very end
            # deliberately: a workflow's command sequence is its replay
            # contract, and adding commands after the existing ones leaves
            # runs that were in flight at deploy replaying cleanly. Nothing in
            # this run consumes the verdict — laser populate does, on the next
            # firing of the stage-0.1 parent — so there is no reason to insert
            # it earlier and every reason not to.
            #
            # CPU queue, not the GPU one that just produced the predictions:
            # it is a line fit, and holding a contended NRP card through it
            # would be waste. That means waking the CPU Deployment, which is
            # cheap and idempotent, and usually already up for the preprocess
            # stages.
            await _dispatch.wake_data_worker()
            summary: LaserAutoAcceptSummary = await _dispatch.dispatch_child(
                "EvaluateLaserAutoAcceptWorkflow",
                dive_id,
                child_id=f"auto-accept-laser-{dive_id}",
                execution_timeout=timedelta(minutes=30),
                result_type=LaserAutoAcceptSummary,
            )
            # Logged at the parent because the per-dive verdict mix is the
            # monitoring signal for this stage — cheaper and faster than the
            # audit sample, and it needs no human labels. Watch BOTH tails: a
            # dive routing far more frames to humans than usual is a detector
            # or an environment that changed, and a suspiciously low flag rate
            # in a new environment is the signature of the one failure
            # consensus cannot self-detect.
            workflow.logger.info(
                "auto-accept gate dive_id=%d eligible=%s reason=%s "
                "auto_accepted=%d/%d verdicts=%s",
                dive_id,
                summary.eligible,
                summary.reason,
                summary.auto_accepted,
                sum(summary.verdicts.values()),
                summary.verdicts,
            )

            if summary.auto_accepted:
                # Apply the verdicts to tasks that already exist. Populate
                # imports an auto-accepted frame already annotated, but it
                # imports a task exactly once — so for a dive still being
                # labeled, which is every dive this cohort selects, the verdict
                # would otherwise change the database and nothing a labeler
                # sees. Same gap #493 fixed for slate predictions and the
                # laser pre-annotation backfill fixed after it.
                #
                # After the gate, necessarily: it reads the verdicts the gate
                # just wrote. Last in the workflow for the same reason the gate
                # is — new commands appended after the existing ones keep
                # in-flight runs replaying — and because its failure leaves
                # nothing to clean up. The verdicts are already persisted, so
                # a failed run here is repaired by the next firing rather than
                # losing anything.
                applied = await _dispatch.run_sdk_activity(
                    "apply_laser_auto_accept_for_dive_activity", dive_id
                )
                workflow.logger.info(
                    "auto-accept applied to existing tasks dive_id=%d applied=%s",
                    dive_id,
                    applied,
                )

        return inputs.dive_id

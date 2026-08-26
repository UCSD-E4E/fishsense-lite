"""Slate-detector parent workflow (api-worker side).

**RETIRED 2026-08-03 — registered, but nothing schedules it.** The
ECC >= 0.80 acceptance gate does not transfer out of distribution: pool
dives produced high-ECC (0.93-0.97) *false* fits that sailed through it
(prod dives 65/71/77/80/83, all pool). The team declined an
active-learning loop; `predict-slate-images-workflow-schedule` is now
actively deleted at worker startup (`worker._RETIRED_SCHEDULE_IDS`) and
the 130 seeded Label Studio predictions were removed.

The code is kept registered so a future evaluation can start it by hand
— it is dormant, not dead — but nothing invokes it on its own. Do not
read it as part of the live pipeline.


Model-assisted slate labeling. Picks the next HIGH-priority dive needing slate
predictions, resolves its unpredicted slate-frame set + template + camera
intrinsics via SDK, stages the raw `.ORF` bytes AND the slate template PDF, and
dispatches the data-worker's `PredictSlateImagesWorkflow`. The child returns
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

Also like the laser-predict parent, it dispatches to
`fishsense_data_processing_gpu_queue`, which means **prefer a GPU, not require
one** — that queue is served by the GPU Deployment or, when it can't start, a
CPU-only one running the same checkpoint. The slate masker does not need a GPU
(fishsense-core measures it at 202 ms/frame on CPU) but goes faster on the card
the laser stage already has, and this way it is never gated on one. When
`wake_gpu_worker` reports that neither Deployment could start, the parent
returns before staging any bytes rather than hanging a child on an unserved
queue.
"""

from datetime import timedelta
from typing import List

from fishsense_shared import PredictSlateImagesInput, SlatePredictionResult
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.activities.gpu_fallback import MODE_UNAVAILABLE

from fishsense_api_workflow_worker.workflows import _dispatch


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
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_slate_prediction_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_slate_predict_inputs_activity",
            dive_id,
            PredictSlateImagesInput,
        )

        workflow.logger.info(
            "dispatching slate predict to data-worker dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.images),
        )

        if not inputs.images:
            return inputs.dive_id

        mode = await _dispatch.wake_gpu_worker()
        if mode == MODE_UNAVAILABLE:
            workflow.logger.warning(
                "no worker available for the slate-predict queue; "
                "skipping dive_id=%d this firing",
                inputs.dive_id,
            )
            return None

        workflow.logger.info("slate predict running on %s capacity", mode)
        await _dispatch.stage_raw(dive_id)
        # The predict activity rectifies the raw frame AND renders the PDF
        # into the template, so both must be staged first.
        await _dispatch.stage_slate_pdf(
            inputs.slate_id,
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=5),
            retry_policy=_dispatch.STAGE_RAW_RETRY_POLICY,
        )
        results: List[SlatePredictionResult] = await _dispatch.dispatch_child(
            "PredictSlateImagesWorkflow",
            inputs,
            child_id=f"predict-slate-{dive_id}",
            # Room for the CPU fallback, which runs the same checkpoint without
            # a GPU. Less of a stretch than the laser stage's — the slate mask
            # is only ~202 ms/frame on CPU.
            execution_timeout=timedelta(hours=4),
            result_type=List[SlatePredictionResult],
            task_queue=_dispatch.DATA_PROCESSING_GPU_TASK_QUEUE,
        )

        if results:
            await _dispatch.run_sdk_activity(
                "persist_slate_predictions_activity", results
            )
            # Attach the freshly-persisted predictions to any *existing*
            # dive-slate LS tasks. Populate seeds pre-annotations only at
            # import time and runs once per dive, so a dive already populated
            # before it was predicted would otherwise never surface them.
            # Idempotent (skips tasks that already carry a prediction).
            await _dispatch.run_sdk_activity(
                "backfill_slate_predictions_for_dive_activity", dive_id
            )

        await _dispatch.cleanup_raw(dive_id)

        return inputs.dive_id

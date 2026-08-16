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

        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        results: List[LaserPredictionResult] = await _dispatch.dispatch_child(
            "PredictLaserImagesWorkflow",
            inputs,
            child_id=f"predict-laser-{dive_id}",
            execution_timeout=timedelta(hours=2),
            result_type=List[LaserPredictionResult],
        )

        if results:
            await _dispatch.run_sdk_activity(
                "persist_laser_predictions_activity", results
            )

        await _dispatch.cleanup_raw(dive_id)

        return inputs.dive_id

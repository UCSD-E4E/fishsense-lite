"""Activity to persist the slate-detector's per-image predictions.

The data-worker `PredictSlateImagesWorkflow` returns one `SlatePredictionResult`
per image; the api-worker parent hands them here to upsert via the SDK
(`put_slate_prediction`, natural-key on image_id). Runs on the api side so the
SDK call stays on the orchestrator's docker network.
"""

from __future__ import annotations

from typing import Any, List

from fishsense_api_sdk.models.slate_prediction import SlatePrediction
from fishsense_shared import SlatePredictionResult
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def persist_slate_predictions_activity(results: List[Any]) -> int:
    """Upsert each prediction; returns the count written."""
    parsed: List[SlatePredictionResult] = [
        r if isinstance(r, SlatePredictionResult)
        else SlatePredictionResult.model_validate(r)
        for r in results
    ]
    activity.logger.info("persisting %d slate predictions", len(parsed))

    async with get_fs_client() as fs:
        for result in parsed:
            await fs.labels.put_slate_prediction(
                result.image_id,
                SlatePrediction(
                    image_id=result.image_id,
                    reference_points=result.reference_points,
                    confidence=result.confidence,
                    rejected_reason=result.rejected_reason,
                    width=result.width,
                    height=result.height,
                ),
            )
    return len(parsed)

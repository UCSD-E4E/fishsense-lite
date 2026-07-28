"""Activity to persist the laser-detector's per-image predictions.

The data-worker `PredictLaserImagesWorkflow` returns one
`LaserPredictionResult` per image; the api-worker parent hands them here to
upsert via the SDK (`put_laser_prediction`, natural-key on image_id). Runs on
the api side so the SDK call stays on the orchestrator's docker network.
"""

from __future__ import annotations

from typing import Any, List

from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_shared import LaserPredictionResult
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def persist_laser_predictions_activity(results: List[Any]) -> int:
    """Upsert each prediction; returns the count written."""
    parsed: List[LaserPredictionResult] = [
        r if isinstance(r, LaserPredictionResult)
        else LaserPredictionResult.model_validate(r)
        for r in results
    ]
    activity.logger.info("persisting %d laser predictions", len(parsed))

    async with get_fs_client() as fs:
        for result in parsed:
            await fs.labels.put_laser_prediction(
                result.image_id,
                LaserPrediction(
                    image_id=result.image_id,
                    x=result.x,
                    y=result.y,
                    confidence=result.confidence,
                ),
            )
    return len(parsed)

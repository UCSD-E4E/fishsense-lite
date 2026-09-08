"""Activity to persist the head/tail predict stage's per-image results.

The data-worker's `PredictHeadtailImagesWorkflow` returns one
`HeadtailPredictionResult` per image; the api-worker parent hands them here to
upsert via the SDK (`put_headtail_prediction`, natural key on image_id). Runs
on the api side so the SDK call stays on the orchestrator's docker network.

Abstentions are persisted too, not dropped. The cohort selects on the row's
absence, so an image the model declined to predict would otherwise be
re-attempted every hour forever; `status` records which kind of abstention it
was, which is also the only way "the model found nothing" stays
distinguishable from "the laser landed on no fish".
"""

from __future__ import annotations

from typing import Any, List

from fishsense_api_sdk.models.head_tail_prediction import HeadTailPrediction
from fishsense_shared.preprocess_contracts import HeadtailPredictionResult
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def persist_headtail_predictions_activity(results: List[Any]) -> int:
    """Upsert each prediction; returns the count written."""
    parsed: List[HeadtailPredictionResult] = [
        (
            r
            if isinstance(r, HeadtailPredictionResult)
            else HeadtailPredictionResult.model_validate(r)
        )
        for r in results
    ]
    activity.logger.info("persisting %d headtail predictions", len(parsed))

    async with get_fs_client() as fs:
        for result in parsed:
            await fs.labels.put_headtail_prediction(
                result.image_id,
                HeadTailPrediction(
                    image_id=result.image_id,
                    head_x=result.head_x,
                    head_y=result.head_y,
                    tail_x=result.tail_x,
                    tail_y=result.tail_y,
                    width=result.width,
                    height=result.height,
                    mask_area_px=result.mask_area_px,
                    silhouette_ratio=result.silhouette_ratio,
                    crop_x=result.crop_x,
                    crop_y=result.crop_y,
                    laser_label_id=result.laser_label_id,
                    predictor_version=result.predictor_version,
                    checkpoint=result.checkpoint,
                    core_version=result.core_version,
                    status=result.status,
                ),
            )
    return len(parsed)

"""Model-assisted laser labeling: fan out `predict_laser_image` across a
dive's laser-unlabeled images and return one prediction per image.

The fishsense-core `LaserDetector` (v2.2.0+) runs on the GPU data-worker.
Predictions seed the laser Label Studio tasks as pre-annotations
(assisted review) — a labeler confirms or nudges each point rather than
placing it from scratch.

Same split as the preprocess stages: the api-worker parent does dive
selection + SDK fetches + raw-byte staging and starts this as a child on
`fishsense_data_processing_queue`. The workflow-level input DTO lives in
`fishsense_shared` (the api-worker / data-worker contract); the per-image
`PredictLaserImageInput` and `LaserPredictionResult` stay here because
they're only constructed inside the fan-out.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import LaserPredictionResult, PredictLaserImagesInput
from pydantic import BaseModel
from temporalio import workflow

__all__ = [
    "LaserPredictionResult",
    "PredictLaserImageInput",
    "PredictLaserImagesWorkflow",
]


class PredictLaserImageInput(BaseModel):
    """Per-image payload passed to the predict_laser_image activity."""

    checksum: str
    image_id: int
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    # Laser wavelength ("red" / "green"), or None when unknown — the model
    # takes an "unknown" wavelength channel in that case.
    wavelength: str | None = None


@workflow.defn
class PredictLaserImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(
        self, payload: PredictLaserImagesInput
    ) -> List[LaserPredictionResult]:
        workflow.logger.info(
            "predicting laser images dive_id=%d images=%d",
            payload.dive_id,
            len(payload.images),
        )

        results = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "predict_laser_image",
                    PredictLaserImageInput(
                        checksum=image.checksum,
                        image_id=image.image_id,
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                        wavelength=payload.wavelength,
                    ),
                    start_to_close_timeout=timedelta(minutes=10),
                    result_type=LaserPredictionResult,
                )
                for image in payload.images
            ]
        )
        return list(results)

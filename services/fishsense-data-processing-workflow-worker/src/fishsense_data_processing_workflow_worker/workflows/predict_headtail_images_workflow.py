"""Model-assisted head/tail labeling: fan out `predict_headtail_image` across a
dive's laser-valid, head/tail-unlabelled images and return one result each.

SAM3 runs on the GPU data-worker. Predictions seed the head/tail Label Studio
tasks as pre-annotations — a labeler confirms or nudges the two keypoints
rather than placing them from scratch.

Thinner than the other predict children in one specific way: there is nothing
to stage. The stage-5.1 JPEG this reads is already in Garage, and it is the
exact frame the labeler is shown, so the parent does no NAS work and the
per-image payload carries no camera intrinsics — only the checksum and the
laser dots, which are both the gate and the crop centre.

Every DTO here lives in `fishsense_shared.preprocess_contracts`, including the
per-image one, because unlike the laser stage nothing is constructed inside the
fan-out: the resolver already emits exactly what each activity needs.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared.preprocess_contracts import (
    HeadtailPredictionResult,
    PredictHeadtailImagesInput,
)
from temporalio import workflow

__all__ = ["PredictHeadtailImagesWorkflow"]


@workflow.defn
class PredictHeadtailImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(
        self, payload: PredictHeadtailImagesInput
    ) -> List[HeadtailPredictionResult]:
        workflow.logger.info(
            "predicting headtail images dive_id=%d images=%d",
            payload.dive_id,
            len(payload.images),
        )

        results = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "predict_headtail_image",
                    image.model_copy(update={"jpeg_folder": payload.jpeg_folder}),
                    # Generous next to the ~0.6 s a prediction takes, because
                    # the first activity on a cold pod also pays for the SAM3
                    # weights: a Garage fetch on a cache miss, then the load.
                    start_to_close_timeout=timedelta(minutes=15),
                    result_type=HeadtailPredictionResult,
                )
                for image in payload.images
            ]
        )
        return list(results)

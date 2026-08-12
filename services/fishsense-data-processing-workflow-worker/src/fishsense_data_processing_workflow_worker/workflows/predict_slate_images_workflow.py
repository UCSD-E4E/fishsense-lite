"""Model-assisted slate labeling: fan out `predict_slate_image` across a
dive's slate frames and return one gated prediction per image.

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

Runs the fishsense-core dive-slate estimator (`fishsense_core.slate`, v2.4.0+)
on the CPU data-worker. Predictions seed the dive-slate Label Studio tasks as
pre-annotations (assisted review) — a labeler confirms or nudges the board
reference points rather than placing all of them from scratch.

Same split as the preprocess/laser-predict stages: an api-worker parent does
dive selection + SDK fetches + raw-byte staging and starts this as a child on
`fishsense_data_processing_queue`. The workflow-level input DTO
(`PredictSlateImagesInput`) and the result (`SlatePredictionResult`) live in
`fishsense_shared` (the cross-worker contract); the per-image
`PredictSlateImageInput` stays here because it's only constructed inside the
fan-out.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import PredictSlateImagesInput, SlatePredictionResult
from pydantic import BaseModel
from temporalio import workflow

__all__ = [
    "PredictSlateImageInput",
    "SlatePredictionResult",
    "PredictSlateImagesInput",
    "PredictSlateImagesWorkflow",
]


class PredictSlateImageInput(BaseModel):
    """Per-image payload passed to the predict_slate_image activity."""

    checksum: str
    image_id: int
    slate_id: int
    slate_name: str
    dpi: float
    # DiveSlate.reference_points — the template's metric reference points.
    template_points: List[List[float]]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


@workflow.defn
class PredictSlateImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(
        self, payload: PredictSlateImagesInput
    ) -> List[SlatePredictionResult]:
        workflow.logger.info(
            "predicting slate images dive_id=%d slate_id=%d images=%d",
            payload.dive_id,
            payload.slate_id,
            len(payload.images),
        )

        results = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "predict_slate_image",
                    PredictSlateImageInput(
                        checksum=image.checksum,
                        image_id=image.image_id,
                        slate_id=payload.slate_id,
                        slate_name=payload.slate_name,
                        dpi=payload.dpi,
                        template_points=payload.template_points,
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                    ),
                    start_to_close_timeout=timedelta(minutes=10),
                    result_type=SlatePredictionResult,
                )
                for image in payload.images
            ]
        )
        return list(results)

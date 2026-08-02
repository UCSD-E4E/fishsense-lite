"""Model-assisted slate labeling: fan out `predict_slate_image` across a dive's
slate frames and return one gated prediction per image.

Runs the fishsense-core dive-slate estimator (`fishsense_core.slate`, v2.4.0+)
on the CPU data-worker. Predictions seed the dive-slate Label Studio tasks as
pre-annotations (assisted review) — a labeler confirms or nudges the board
reference points rather than placing all of them from scratch.

Same split as the preprocess/laser-predict stages: an api-worker parent does
dive selection + SDK fetches + raw-byte staging and starts this as a child on
`fishsense_data_processing_queue`. The per-image `PredictSlateImageInput` and
`SlatePredictionResult` live here because they're only constructed inside the
fan-out; the parent-facing `PredictSlateImagesInput` will move to
`fishsense_shared` when the parent lands (kept local until then).
"""

import asyncio
from datetime import timedelta
from typing import List

from pydantic import BaseModel
from temporalio import workflow

__all__ = [
    "SlateImageRef",
    "PredictSlateImageInput",
    "SlatePredictionResult",
    "PredictSlateImagesInput",
    "PredictSlateImagesWorkflow",
]


class SlateImageRef(BaseModel):
    """One slate frame to predict: its raw checksum + image id."""

    checksum: str
    image_id: int


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


class SlatePredictionResult(BaseModel):
    """One frame's gated prediction, or the reason there isn't one.

    `reference_points` are in rectified-photo pixels (the same space the sync
    activity stores `DiveSlateLabel.reference_points` after stripping the
    composite panel offset); the slate populate step converts them to composite
    Label Studio coordinates. `None` when the estimate was rejected — see
    `rejected_reason` (`unsupported_slate_family` / `no_board` /
    `low_confidence` / `points_off_canvas`).
    """

    image_id: int
    reference_points: List[List[float]] | None = None
    confidence: float = 0.0
    rejected_reason: str | None = None
    width: int = 0
    height: int = 0


class PredictSlateImagesInput(BaseModel):
    """Parent-built workflow input: the dive's slate template + frames."""

    dive_id: int
    slate_id: int
    slate_name: str
    dpi: float
    template_points: List[List[float]]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    images: List[SlateImageRef]


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

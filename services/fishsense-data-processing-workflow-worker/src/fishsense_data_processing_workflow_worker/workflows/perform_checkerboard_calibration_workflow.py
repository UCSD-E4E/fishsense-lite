"""Fan out checkerboard detection across a dive's frames, then fit its laser.

The data-worker child of `PerformCheckerboardLaserCalibrationParentWorkflow`.
Two phases, because the two halves have different costs: finding the board is
per-frame and holds image bytes (a rawpy decode peaking at 1-3 GB), while the
fit is a few milliseconds of arithmetic over a list of 3-D points.

Same split as the preprocess stages — the api-worker parent does dive
selection, SDK fetches and raw-byte staging and starts this as a child on
`fishsense_data_processing_queue`. The workflow-level input DTO lives in
`fishsense_shared` (the api-worker / data-worker contract); the per-image
`DetectCheckerboardLaserPointInput` and the fit's input stay here because they
are only constructed inside the fan-out.

Note this runs on the **CPU** queue, unlike stage 13's calibration child which
runs on the light queue. Stage 13 reads already-stored slate labels and never
touches an image; this one decodes every calibration frame.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import (
    CheckerboardObservation,
    PerformCheckerboardCalibrationInput,
)
from pydantic import BaseModel
from temporalio import workflow

__all__ = [
    "DetectCheckerboardLaserPointInput",
    "FitCheckerboardExtrinsicsInput",
    "PerformCheckerboardCalibrationWorkflow",
]


class DetectCheckerboardLaserPointInput(BaseModel):
    """Per-image payload for the `detect_checkerboard_laser_point` activity."""

    image_id: int
    checksum: str
    laser_x: float
    laser_y: float
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    # The declared board's INTERIOR corners — an upper bound on what may be
    # detected, not the grid asked for. See `checkerboard_detection`.
    target_rows: int
    target_cols: int
    square_size_m: float


class FitCheckerboardExtrinsicsInput(BaseModel):
    """Dive-level payload for the `fit_checkerboard_laser_extrinsics` activity."""

    dive_id: int
    camera_id: int
    camera_matrix: List[List[float]]
    observations: List[CheckerboardObservation]


@workflow.defn
class PerformCheckerboardCalibrationWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PerformCheckerboardCalibrationInput) -> int:
        workflow.logger.info(
            "checkerboard calibration dive_id=%d frames=%d board=%dx%d pitch=%.5fm",
            payload.dive_id,
            len(payload.images),
            payload.target_rows,
            payload.target_cols,
            payload.square_size_m,
        )

        observations = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "detect_checkerboard_laser_point",
                    DetectCheckerboardLaserPointInput(
                        image_id=image.image_id,
                        checksum=image.checksum,
                        laser_x=image.laser_x,
                        laser_y=image.laser_y,
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                        target_rows=payload.target_rows,
                        target_cols=payload.target_cols,
                        square_size_m=payload.square_size_m,
                    ),
                    start_to_close_timeout=timedelta(minutes=10),
                    result_type=CheckerboardObservation,
                )
                for image in payload.images
            ]
        )

        # One activity for the fit, not inline in the workflow: `calibrate_laser`
        # is a Rust kernel and `check_fit_self_consistency` is numpy, neither of
        # which may run in workflow code — a workflow body has to be
        # deterministic and replayable.
        return await workflow.execute_activity(
            "fit_checkerboard_laser_extrinsics",
            FitCheckerboardExtrinsicsInput(
                dive_id=payload.dive_id,
                camera_id=payload.camera_id,
                camera_matrix=payload.camera_matrix,
                observations=list(observations),
            ),
            start_to_close_timeout=timedelta(minutes=10),
        )

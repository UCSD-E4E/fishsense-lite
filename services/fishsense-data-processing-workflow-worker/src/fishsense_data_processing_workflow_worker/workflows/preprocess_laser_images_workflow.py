"""Stage 0.1 workflow: pick the next HIGH-priority dive needing laser
preprocessing, resolve its incomplete-laser-label image set, and fan
out per-image rectify/overlay/encode work.

The selector + resolver activities live alongside the per-image activity
on the data-worker — the data-worker already uses the SDK for stages
13/14, so the api-worker stays thin (no orchestrator).
"""

import asyncio
from datetime import timedelta
from typing import List, Tuple

from pydantic import BaseModel
from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_data_processing_workflow_worker.activities.resolve_laser_preprocess_inputs_activity import (  # noqa: E501  pylint: disable=line-too-long
        LaserPreprocessInputs,
    )

# Notebook hardcoded; promoted to a workflow constant when the input
# layer was dropped. Move to settings if a second bbox is ever needed.
DEFAULT_LASER_BBOX: Tuple[int, int, int, int] = (1800, 700, 2400, 1600)
OUTPUT_FOLDER = "preprocess_jpeg"


class PreprocessLaserImageInput(BaseModel):
    """Per-image payload passed to the preprocess_laser_image activity."""

    checksum: str
    output_folder: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in rectified pixels
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


@workflow.defn
class PreprocessLaserImagesWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive without laser extrinsics and
    preprocess every still-incomplete laser image for it.

    Returns the dive_id processed (or None when the backlog is empty).
    Idempotent: each invocation drains exactly one dive, so an hourly
    schedule clears an N-dive backlog in N hours. Operators can also
    trigger ad-hoc runs via `temporal workflow start`.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await workflow.execute_activity(
            "select_next_high_priority_dive_for_laser_preprocessing_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        if dive_id is None:
            return None

        inputs: LaserPreprocessInputs = await workflow.execute_activity(
            "resolve_laser_preprocess_inputs_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
            result_type=LaserPreprocessInputs,
        )

        workflow.logger.info(
            "preprocessing laser images dive_id=%d images=%d",
            inputs.dive_id,
            len(inputs.image_checksums),
        )

        if not inputs.image_checksums:
            return inputs.dive_id

        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "preprocess_laser_image",
                    PreprocessLaserImageInput(
                        checksum=checksum,
                        output_folder=OUTPUT_FOLDER,
                        bbox=DEFAULT_LASER_BBOX,
                        camera_matrix=inputs.camera_matrix,
                        distortion_coefficients=inputs.distortion_coefficients,
                    ),
                    schedule_to_close_timeout=timedelta(minutes=5),
                )
                for checksum in inputs.image_checksums
            ]
        )

        return inputs.dive_id

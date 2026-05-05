"""Stage 0.1 workflow: fan out preprocess_laser_image across every
incomplete laser-labeled image of a dive.

Inputs are pre-resolved by the api-worker parent
(`PreprocessLaserImagesParentWorkflow` on `fishsense_api_queue`),
which does dive selection + SDK fetches + raw-byte staging and then
starts this workflow as a child on `fishsense_data_processing_queue`.
This workflow does not call fishsense-api, the NAS, or the
file-exchange itself — it only orchestrates per-image activities.

The workflow-level input DTO `PreprocessLaserImagesInput` lives in
`fishsense_shared` because it's the api-worker / data-worker
contract; the per-image `PreprocessLaserImageInput` stays here
because it's only constructed inside the fan-out.
"""

import asyncio
from datetime import timedelta
from typing import List, Tuple

from fishsense_shared import PreprocessLaserImagesInput
from pydantic import BaseModel
from temporalio import workflow


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
    @workflow.run
    async def run(self, payload: PreprocessLaserImagesInput) -> None:
        workflow.logger.info(
            "preprocessing laser images dive_id=%d images=%d",
            payload.dive_id,
            len(payload.image_checksums),
        )

        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "preprocess_laser_image",
                    PreprocessLaserImageInput(
                        checksum=checksum,
                        output_folder="preprocess_jpeg",
                        bbox=tuple(payload.bbox),
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                )
                for checksum in payload.image_checksums
            ]
        )

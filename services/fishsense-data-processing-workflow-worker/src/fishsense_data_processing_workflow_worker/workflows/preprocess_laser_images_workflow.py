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
from typing import List, Optional, Tuple

from fishsense_shared import PreprocessLaserImagesInput
from pydantic import BaseModel
from temporalio import workflow


class PreprocessLaserImageInput(BaseModel):
    """Per-image payload passed to the preprocess_laser_image activity.

    `region` is the shape actually drawn; `bbox` is its bounding box and the
    fallback when the parent's payload predates the polygon. Both are carried
    for the same reason `PreprocessLaserImagesInput` carries both -- see that
    DTO -- and `bbox` stays required so a child workflow already in flight at
    deploy replays against an unchanged activity input.
    """

    checksum: str
    output_folder: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in rectified pixels
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    region: Optional[List[List[int]]] = None  # [[x, y], ...] rectified pixels


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
                        region=payload.laser_region,
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                )
                for checksum in payload.image_checksums
            ]
        )

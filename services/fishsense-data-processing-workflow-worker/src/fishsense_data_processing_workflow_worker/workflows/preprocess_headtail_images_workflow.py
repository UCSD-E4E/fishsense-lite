"""Stage 5.1 workflow: fan out preprocess_headtail_image across every
image whose head/tail label is incomplete for a dive.

Inputs are pre-resolved by the api-worker parent
(`PreprocessHeadtailImagesParentWorkflow` on `fishsense_api_queue`),
which does dive selection + SDK fetches and then starts this workflow
as a child on `fishsense_data_processing_queue`.

The workflow-level input DTO `PreprocessHeadtailImagesInput` lives in
`fishsense_shared` because it's the api-worker / data-worker contract.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import PreprocessHeadtailImagesInput
from pydantic import BaseModel
from temporalio import workflow


class PreprocessHeadtailImageInput(BaseModel):
    """Per-image payload passed to the preprocess_headtail_image activity."""

    checksum: str
    output_folder: str
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


@workflow.defn
class PreprocessHeadtailImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PreprocessHeadtailImagesInput) -> None:
        workflow.logger.info(
            "preprocessing headtail images dive_id=%d images=%d",
            payload.dive_id,
            len(payload.image_checksums),
        )

        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "preprocess_headtail_image",
                    PreprocessHeadtailImageInput(
                        checksum=checksum,
                        output_folder="preprocess_headtail_jpeg",
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                )
                for checksum in payload.image_checksums
            ]
        )

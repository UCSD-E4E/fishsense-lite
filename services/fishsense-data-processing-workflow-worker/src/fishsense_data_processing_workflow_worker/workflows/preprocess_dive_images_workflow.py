"""Stage 2 workflow: fan out preprocess_dive_image across every image
in every cluster of a dive.

Inputs are pre-resolved by the api-worker parent
(`PreprocessDiveImagesParentWorkflow` on `fishsense_api_queue`),
which does dive selection + SDK fetches and then starts this workflow
as a child on `fishsense_data_processing_queue`. This workflow does
not call fishsense-api, the NAS, or the file-exchange itself — it
only orchestrates per-image activities.

The workflow-level input DTO `PreprocessDiveImagesInput` lives in
`fishsense_shared` because it's the api-worker / data-worker
contract.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import PreprocessDiveImagesInput
from pydantic import BaseModel
from temporalio import workflow


class PreprocessDiveImageInput(BaseModel):
    """Per-image input passed to the preprocess_dive_image activity."""

    checksum: str
    cluster_index: int  # 1-based
    cluster_size: int
    output_folder: str
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


@workflow.defn
class PreprocessDiveImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PreprocessDiveImagesInput) -> None:
        workflow.logger.info(
            "preprocessing dive_id=%d clusters=%d images=%d",
            payload.dive_id,
            len(payload.clusters),
            sum(len(c) for c in payload.clusters),
        )

        for cluster in payload.clusters:
            await asyncio.gather(
                *[
                    workflow.execute_activity(
                        "preprocess_dive_image",
                        PreprocessDiveImageInput(
                            checksum=checksum,
                            cluster_index=i + 1,
                            cluster_size=len(cluster),
                            output_folder="preprocess_groups_jpeg",
                            camera_matrix=payload.camera_matrix,
                            distortion_coefficients=payload.distortion_coefficients,
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                    )
                    for i, checksum in enumerate(cluster)
                ]
            )

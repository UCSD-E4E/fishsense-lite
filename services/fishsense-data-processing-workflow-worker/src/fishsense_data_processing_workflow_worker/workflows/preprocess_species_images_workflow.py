"""Stage 2 workflow: fan out preprocess_species_image across every image
in every cluster of a dive.

Inputs are pre-resolved by the api-worker parent
(`PreprocessSpeciesImagesParentWorkflow` on `fishsense_api_queue`),
which does dive selection + SDK fetches and then starts this workflow
as a child on `fishsense_data_processing_queue`. Cluster image_ids
are pre-filtered by the api-worker resolver to images with a valid
laser label and no non-sentinel species label, so the cluster
numbering ("image i of N") reflects the labeler-visible subset.

This workflow does not call fishsense-api, the NAS, or the
file-exchange itself — it only orchestrates per-image activities.

The workflow-level input DTO `PreprocessSpeciesImagesInput` lives in
`fishsense_shared` because it's the api-worker / data-worker
contract.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import PreprocessSpeciesImagesInput
from pydantic import BaseModel
from temporalio import workflow


class PreprocessSpeciesImageInput(BaseModel):
    """Per-image input passed to the preprocess_species_image activity."""

    checksum: str
    cluster_index: int  # 1-based
    cluster_size: int
    output_folder: str
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


@workflow.defn
class PreprocessSpeciesImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PreprocessSpeciesImagesInput) -> None:
        workflow.logger.info(
            "preprocessing species dive_id=%d clusters=%d images=%d",
            payload.dive_id,
            len(payload.clusters),
            sum(len(c) for c in payload.clusters),
        )

        for cluster in payload.clusters:
            await asyncio.gather(
                *[
                    workflow.execute_activity(
                        "preprocess_species_image",
                        PreprocessSpeciesImageInput(
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

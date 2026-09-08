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

from fishsense_shared import PreprocessSpeciesImagesInput, SpeciesClusterMember
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

        # `cluster_members` carries each frame's position in the WHOLE
        # PREDICTION cluster. Numbering `clusters` positionally instead is only
        # right when every member of a cluster is being redrawn at once: the
        # resolver emits just the images that need work, so a partial pass --
        # the normal case under `needs_reprocess`, and also whenever an image
        # becomes eligible after its cluster was first processed -- would label
        # 3 frames of a 7-image cluster "1 of 3".."3 of 3" while their siblings
        # still read "4 of 7".."7 of 7" at the same object-store keys.
        #
        # The fallback keeps this workflow able to run a payload from an older
        # api-worker, which sends no `cluster_members` at all; it reproduces the
        # previous numbering exactly rather than inventing something new.
        groups = payload.cluster_members
        if groups is None:
            groups = [
                [
                    SpeciesClusterMember(
                        checksum=checksum,
                        cluster_index=i + 1,
                        cluster_size=len(cluster),
                    )
                    for i, checksum in enumerate(cluster)
                ]
                for cluster in payload.clusters
            ]

        for group in groups:
            await asyncio.gather(
                *[
                    workflow.execute_activity(
                        "preprocess_species_image",
                        PreprocessSpeciesImageInput(
                            checksum=member.checksum,
                            cluster_index=member.cluster_index,
                            cluster_size=member.cluster_size,
                            output_folder="preprocess_groups_jpeg",
                            camera_matrix=payload.camera_matrix,
                            distortion_coefficients=payload.distortion_coefficients,
                        ),
                        start_to_close_timeout=timedelta(minutes=5),
                    )
                    for member in group
                ]
            )

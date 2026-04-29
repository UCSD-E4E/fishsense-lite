"""Stage 2 workflow: fan out preprocess_dive_image across every image
in every cluster of a dive.

Inputs are pre-resolved by the caller (the api-worker, in production):
the workflow does not call fishsense-api, the NAS, or the file-exchange
itself — it only orchestrates activities. Raw `.ORF` bytes must already
have been staged on the file-exchange under
`/api/v1/exchange/raw/{checksum}.ORF` before the workflow runs."""

import asyncio
from datetime import timedelta
from typing import List

from pydantic import BaseModel
from temporalio import workflow

# Note: this stage's parent (which fetches dive/clusters/intrinsics from
# fishsense-api and stages raw .ORFs to the file-exchange) lives in the
# api-worker and is intentionally not part of this workflow. Wire it up
# when porting the api-worker side.


class PreprocessDiveImageInput(BaseModel):
    """Per-image input passed to the preprocess_dive_image activity."""

    checksum: str
    cluster_index: int  # 1-based
    cluster_size: int
    output_folder: str
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


class PreprocessDiveImagesInput(BaseModel):
    """Whole-workflow input: clusters of image checksums plus the
    camera intrinsics needed to rectify each one."""

    dive_id: int
    clusters: List[List[str]]  # each inner list is a cluster of checksums in order
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
                        schedule_to_close_timeout=timedelta(minutes=5),
                    )
                    for i, checksum in enumerate(cluster)
                ]
            )

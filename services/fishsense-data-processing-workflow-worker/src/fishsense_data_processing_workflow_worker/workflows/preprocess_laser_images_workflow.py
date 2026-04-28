"""Stage 0.1 workflow: fan out preprocess_laser_image across every
incomplete laser-labeled image of a dive.

Inputs are pre-resolved by the api-worker (image checksums of incomplete
laser labels for the dive, camera intrinsics, the laser-bbox constant).
The workflow does not call fishsense-api, the NAS, or the file-exchange
itself — it only orchestrates activities. Raw `.ORF` bytes must already
have been staged on the file-exchange under
`/api/v1/exchange/raw/{checksum}.ORF` before the workflow runs.
"""

import asyncio
from datetime import timedelta
from typing import List, Tuple

from pydantic import BaseModel
from temporalio import workflow


class PreprocessLaserImageInput(BaseModel):
    """Per-image input passed to the preprocess_laser_image activity."""

    checksum: str
    output_folder: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in rectified pixels
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


class PreprocessLaserImagesInput(BaseModel):
    """Whole-workflow input: a flat list of image checksums plus the
    camera intrinsics and laser-bbox constant."""

    dive_id: int
    image_checksums: List[str]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    # Original notebook hardcoded (1800, 700, 2400, 1600); kept as input so
    # the api-worker can promote it to config without touching this code.
    bbox: List[int]


@workflow.defn
class PreprocessLaserImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, input: PreprocessLaserImagesInput) -> None:
        workflow.logger.info(
            "preprocessing laser images dive_id=%d images=%d",
            input.dive_id,
            len(input.image_checksums),
        )

        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "preprocess_laser_image",
                    PreprocessLaserImageInput(
                        checksum=checksum,
                        output_folder="preprocess_jpeg",
                        bbox=tuple(input.bbox),
                        camera_matrix=input.camera_matrix,
                        distortion_coefficients=input.distortion_coefficients,
                    ),
                    schedule_to_close_timeout=timedelta(minutes=5),
                )
                for checksum in input.image_checksums
            ]
        )

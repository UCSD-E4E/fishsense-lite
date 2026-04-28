"""Stage 9 workflow: fan out preprocess_slate_image across slate-labeled
images of a dive.

Inputs are pre-resolved by the api-worker. Raw `.ORF` bytes must be
staged on the file-exchange under
`/api/v1/exchange/raw/{checksum}.ORF`, and the slate template PDF under
`/api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf`, before the workflow
runs."""

import asyncio
from datetime import timedelta
from typing import List, Tuple

from pydantic import BaseModel
from temporalio import workflow


ReferencePoint = Tuple[float, float]


class PreprocessSlateImageInput(BaseModel):
    """Per-image input passed to the preprocess_slate_image activity."""

    checksum: str
    output_folder: str
    slate_id: int
    slate_dpi: int
    reference_points: List[ReferencePoint]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


class PreprocessSlateImagesInput(BaseModel):
    """Whole-workflow input."""

    dive_id: int
    image_checksums: List[str]
    slate_id: int
    slate_dpi: int
    reference_points: List[ReferencePoint]
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]


@workflow.defn
class PreprocessSlateImagesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, input: PreprocessSlateImagesInput) -> None:
        workflow.logger.info(
            "preprocessing slate images dive_id=%d images=%d slate_id=%d",
            input.dive_id,
            len(input.image_checksums),
            input.slate_id,
        )

        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "preprocess_slate_image",
                    PreprocessSlateImageInput(
                        checksum=checksum,
                        output_folder="preprocess_slate_images_jpeg",
                        slate_id=input.slate_id,
                        slate_dpi=input.slate_dpi,
                        reference_points=input.reference_points,
                        camera_matrix=input.camera_matrix,
                        distortion_coefficients=input.distortion_coefficients,
                    ),
                    schedule_to_close_timeout=timedelta(minutes=5),
                )
                for checksum in input.image_checksums
            ]
        )

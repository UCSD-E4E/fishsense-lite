"""Stage 2: rectify a raw dive image, overlay the cluster index, and
push the JPEG to the file-exchange.

The pure-logic core (`overlay_and_encode_jpeg`) is broken out as a
module-level function so it's unit-testable without Temporal, httpx, or
fishsense-core decoding."""

import asyncio
import logging

import cv2
import httpx
import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
from temporalio import activity

from fishsense_data_processing_workflow_worker.config import settings
from fishsense_data_processing_workflow_worker.file_exchange import (
    FileExchangeClient,
)

_log = logging.getLogger(__name__)


def overlay_and_encode_jpeg(
    rectified_bgr: np.ndarray,
    cluster_index: int,
    cluster_size: int,
) -> bytes:
    """Draw the 1-based cluster index in the bottom-right corner and
    encode to JPEG bytes. Does not mutate the input."""
    img = rectified_bgr.copy()
    height, width = img.shape[:2]
    cv2.putText(
        img,
        f"{cluster_index}/{cluster_size}",
        (width - 350, height - 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        5,
        (0, 0, 255),
        10,
        cv2.LINE_AA,
    )
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def _rectify_overlay_encode(
    raw_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
    cluster_index: int,
    cluster_size: int,
) -> bytes:
    """Sync helper run via asyncio.to_thread — heavy CPU work
    (rawpy decode + cv2.undistort + skimage CLAHE)."""
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(camera_matrix, dtype=float),
        distortion_coefficients=np.array(distortion_coefficients, dtype=float),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics)
    return overlay_and_encode_jpeg(rectified.data, cluster_index, cluster_size)


# Imported lazily to keep this module importable without the workflow
# package side-effects during unit-test collection.
def _input_model():
    from fishsense_data_processing_workflow_worker.workflows.preprocess_dive_images_workflow import (
        PreprocessDiveImageInput,
    )

    return PreprocessDiveImageInput


@activity.defn
async def preprocess_dive_image(input) -> None:  # type: ignore[no-untyped-def]
    """Download one raw image from the file-exchange, rectify it,
    overlay the cluster index, and PUT the JPEG back to the
    file-exchange under `{output_folder}/{checksum}.JPG`."""
    # Late-bind the model so tests that don't import the workflow module
    # can still construct the activity registration.
    PreprocessDiveImageInput = _input_model()
    if not isinstance(input, PreprocessDiveImageInput):
        input = PreprocessDiveImageInput.model_validate(input)

    activity.logger.info(
        "preprocessing image checksum=%s cluster=%d/%d",
        input.checksum,
        input.cluster_index,
        input.cluster_size,
    )

    async with httpx.AsyncClient(
        base_url=settings.static_file_server.url, timeout=httpx.Timeout(60.0)
    ) as http:
        client = FileExchangeClient(
            base_url=settings.static_file_server.url, http=http
        )
        raw_bytes = await client.download_raw(input.checksum)
        jpeg_bytes = await asyncio.to_thread(
            _rectify_overlay_encode,
            raw_bytes,
            input.camera_matrix,
            input.distortion_coefficients,
            input.cluster_index,
            input.cluster_size,
        )
        await client.upload_processed_jpeg(
            folder=input.output_folder,
            checksum=input.checksum,
            data=jpeg_bytes,
        )

"""Stage 0.1: rectify a raw laser image, draw the expected-laser bounding
box, and PUT the JPEG to the file-exchange.

The pure-logic core (`overlay_laser_bbox_and_encode_jpeg`) is broken out
as a module-level function so it's unit-testable without Temporal,
httpx, or rawpy.
"""

import asyncio
import logging
from typing import Tuple

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


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def overlay_laser_bbox_and_encode_jpeg(
    rectified_bgr: np.ndarray,
    bbox: Bbox,
) -> bytes:
    """Draw a 2-px green rectangle at `bbox` and encode the result to
    JPEG bytes. Does not mutate the input."""
    img = rectified_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def _rectify_overlay_bbox_encode(
    raw_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
    bbox: Bbox,
) -> bytes:
    """Sync helper run via asyncio.to_thread — heavy CPU work
    (rawpy decode + cv2.undistort + skimage CLAHE)."""
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(camera_matrix, dtype=float),
        distortion_coefficients=np.array(distortion_coefficients, dtype=float),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics)
    return overlay_laser_bbox_and_encode_jpeg(rectified.data, bbox)


def _input_model():
    from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (
        PreprocessLaserImageInput,
    )

    return PreprocessLaserImageInput


@activity.defn
async def preprocess_laser_image(input) -> None:  # type: ignore[no-untyped-def]
    """Download one raw image from the file-exchange, rectify it, draw
    the expected-laser bounding box, and PUT the JPEG back to the
    file-exchange under `{output_folder}/{checksum}.JPG`."""
    PreprocessLaserImageInput = _input_model()
    if not isinstance(input, PreprocessLaserImageInput):
        input = PreprocessLaserImageInput.model_validate(input)

    activity.logger.info(
        "preprocessing laser image checksum=%s bbox=%s",
        input.checksum,
        input.bbox,
    )

    async with httpx.AsyncClient(
        base_url=settings.static_file_server.url, timeout=httpx.Timeout(60.0)
    ) as http:
        client = FileExchangeClient(
            base_url=settings.static_file_server.url, http=http
        )
        raw_bytes = await client.download_raw(input.checksum)
        jpeg_bytes = await asyncio.to_thread(
            _rectify_overlay_bbox_encode,
            raw_bytes,
            input.camera_matrix,
            input.distortion_coefficients,
            tuple(input.bbox),
        )
        await client.upload_processed_jpeg(
            folder=input.output_folder,
            checksum=input.checksum,
            data=jpeg_bytes,
        )

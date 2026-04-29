"""Stage 5.1: rectify a raw head/tail image and PUT the JPEG to the
file-exchange. No overlay — head/tail labeling wants the unannotated
rectified frame.
"""

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


def encode_rectified_jpeg(rectified_bgr: np.ndarray) -> bytes:
    """Encode a rectified BGR ndarray to JPEG bytes. Does not mutate."""
    success, encoded = cv2.imencode(".jpg", rectified_bgr)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def _rectify_and_encode_jpeg(
    raw_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
) -> bytes:
    """Sync helper run via asyncio.to_thread."""
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(camera_matrix, dtype=float),
        distortion_coefficients=np.array(distortion_coefficients, dtype=float),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics)
    return encode_rectified_jpeg(rectified.data)


def _input_model():
    # pylint: disable=import-outside-toplevel
    from fishsense_data_processing_workflow_worker.workflows.preprocess_headtail_images_workflow \
        import PreprocessHeadtailImageInput

    return PreprocessHeadtailImageInput


@activity.defn
async def preprocess_headtail_image(payload) -> None:  # type: ignore[no-untyped-def]
    """Download one raw image, rectify it, and PUT the JPEG to the
    file-exchange under `{output_folder}/{checksum}.JPG`."""
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    activity.logger.info(
        "preprocessing headtail image checksum=%s", payload.checksum
    )

    async with httpx.AsyncClient(
        base_url=settings.static_file_server.url, timeout=httpx.Timeout(60.0)
    ) as http:
        client = FileExchangeClient(
            base_url=settings.static_file_server.url, http=http
        )
        raw_bytes = await client.download_raw(payload.checksum)
        jpeg_bytes = await asyncio.to_thread(
            _rectify_and_encode_jpeg,
            raw_bytes,
            payload.camera_matrix,
            payload.distortion_coefficients,
        )
        await client.upload_processed_jpeg(
            folder=payload.output_folder,
            checksum=payload.checksum,
            data=jpeg_bytes,
        )

"""Stage 9: composite a binarized slate-template PDF render with the
rectified raw image, draw reference-point markers, and PUT the JPEG to
the file-exchange.

Pure-logic helpers (`render_slate_pdf_to_binarized_bgr`,
`composite_slate_with_image`) are module-level so they can be tested
without the Temporal/httpx surface."""

import asyncio
import logging
from typing import List, Tuple

import cv2
import httpx
import numpy as np
import pymupdf
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
from temporalio import activity

from fishsense_data_processing_workflow_worker.config import settings
from fishsense_data_processing_workflow_worker.file_exchange import (
    FileExchangeClient,
)

_log = logging.getLogger(__name__)


ReferencePoint = Tuple[float, float]


def render_slate_pdf_to_binarized_bgr(pdf_bytes: bytes, dpi: int) -> np.ndarray:
    """Render page 0 of a slate template PDF at the given DPI, threshold
    at 125, and return a 3-channel BGR uint8 array — same shape the
    notebook produced via pymupdf -> grayscale -> threshold -> BGR."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page: pymupdf.Page = document.load_page(0)
        pixmap: pymupdf.Pixmap = page.get_pixmap(dpi=dpi)
        raw = np.frombuffer(pixmap.samples, dtype=np.uint8)
        rgb = raw.reshape(pixmap.height, pixmap.width, pixmap.n)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, binarized = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)


def composite_slate_with_image(
    pdf_image: np.ndarray,
    rectified_image: np.ndarray,
    reference_points: List[ReferencePoint],
) -> np.ndarray:
    """Scale the slate render to match the rectified image height,
    horizontally concat (slate left, image right), and overlay each
    reference point as a red filled circle + numbered label.

    Coordinates in `reference_points` are in the original PDF pixel
    space and are scaled by `image_height / pdf_height` to land on the
    final canvas. Mirrors the original notebook's per-image cell
    exactly."""
    img_height, img_width = rectified_image.shape[:2]
    pdf_height, pdf_width = pdf_image.shape[:2]

    scale_y = float(img_height) / float(pdf_height)
    new_pdf_height = int(pdf_height * scale_y)
    new_pdf_width = int(pdf_width * scale_y)
    pdf_resized = cv2.resize(pdf_image, (new_pdf_width, new_pdf_height))

    canvas = np.zeros(
        (img_height, img_width + new_pdf_width, 3), dtype=np.uint8
    )
    canvas[:, :new_pdf_width, :] = pdf_resized
    canvas[:, new_pdf_width:, :] = rectified_image

    for idx, (px, py) in enumerate(reference_points):
        x = int(px * scale_y)
        y = int(py * scale_y)
        cv2.circle(canvas, (x, y), radius=25, color=(0, 0, 255), thickness=-1)
        cv2.putText(
            canvas,
            f"{idx + 1}",
            (x + 20, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            (0, 0, 255),
            10,
            cv2.LINE_AA,
        )

    return canvas


def _build_slate_jpeg(
    raw_bytes: bytes,
    pdf_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
    slate_dpi: int,
    reference_points: List[ReferencePoint],
) -> bytes:
    """Sync helper run via asyncio.to_thread."""
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(camera_matrix, dtype=float),
        distortion_coefficients=np.array(distortion_coefficients, dtype=float),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics).data
    pdf_image = render_slate_pdf_to_binarized_bgr(pdf_bytes, dpi=slate_dpi)
    composite = composite_slate_with_image(
        pdf_image, rectified, reference_points
    )
    success, encoded = cv2.imencode(".jpg", composite)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def _input_model():
    # pylint: disable=import-outside-toplevel
    from fishsense_data_processing_workflow_worker.workflows.preprocess_slate_images_workflow \
        import PreprocessSlateImageInput

    return PreprocessSlateImageInput


@activity.defn
async def preprocess_slate_image(payload) -> None:  # type: ignore[no-untyped-def]
    """Download one raw image + the slate-template PDF, build the slate
    composite, and PUT the JPEG back to the file-exchange under
    `{output_folder}/{checksum}.JPG`."""
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    activity.logger.info(
        "preprocessing slate image checksum=%s slate_id=%d",
        payload.checksum,
        payload.slate_id,
    )

    async with httpx.AsyncClient(
        base_url=settings.file_exchange.url, timeout=httpx.Timeout(60.0)
    ) as http:
        client = FileExchangeClient(
            base_url=settings.file_exchange.url, http=http
        )
        raw_bytes = await client.download_raw(payload.checksum)
        pdf_bytes = await client.download_slate_pdf(payload.slate_id)
        jpeg_bytes = await asyncio.to_thread(
            _build_slate_jpeg,
            raw_bytes,
            pdf_bytes,
            payload.camera_matrix,
            payload.distortion_coefficients,
            payload.slate_dpi,
            [tuple(p) for p in payload.reference_points],
        )
        await client.upload_processed_jpeg(
            folder=payload.output_folder,
            checksum=payload.checksum,
            data=jpeg_bytes,
        )

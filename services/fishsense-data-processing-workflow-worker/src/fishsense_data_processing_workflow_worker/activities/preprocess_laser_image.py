"""Stage 0.1: rectify a raw laser image, draw the expected-laser region,
and write the JPEG to the Garage object store.

The region is a convex polygon as of 2026-08-27 (`LASER_REGION_POLYGON` on
the api-worker, which is where it is derived and pinned). It used to be an
axis-aligned rectangle, and that path is still here: the two workers deploy
independently, so a payload from an api-worker that predates the polygon
arrives with `region=None` and has to keep rendering. The rectangle path is
also what `test_stage0_1_notebook_parity.py` holds byte-for-byte against the
original notebook, so it is deliberately left exactly as it was rather than
re-expressed as a 4-vertex polygon -- `cv2.rectangle` and `cv2.polylines` do
not agree pixel-for-pixel at the corners.

The pure-logic cores (`overlay_laser_region_and_encode_jpeg`,
`overlay_laser_bbox_and_encode_jpeg`) are broken out as module-level
functions so they're unit-testable without Temporal, S3, or rawpy.
"""

import asyncio
import logging
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
from temporalio import activity

from fishsense_data_processing_workflow_worker.object_store import (
    open_object_store_client,
)

_log = logging.getLogger(__name__)


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
Region = Sequence[Sequence[int]]  # [[x, y], ...] convex, in draw order


def _encode_jpeg(img: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("cv2.imencode failed")
    return encoded.tobytes()


def overlay_laser_region_and_encode_jpeg(
    rectified_bgr: np.ndarray,
    region: Region,
) -> bytes:
    """Draw a 2-px green closed outline through `region` and encode the
    result to JPEG bytes. Does not mutate the input.

    An outline, never a fill: the labeler has to see the image under it.
    """
    img = rectified_bgr.copy()
    points = np.asarray(region, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [points], True, (0, 255, 0), 2)
    return _encode_jpeg(img)


def overlay_laser_bbox_and_encode_jpeg(
    rectified_bgr: np.ndarray,
    bbox: Bbox,
) -> bytes:
    """Draw a 2-px green rectangle at `bbox` and encode the result to
    JPEG bytes. Does not mutate the input.

    The pre-polygon shape, kept as the version-skew fallback. See the module
    docstring for why it is not expressed in terms of the polygon path.
    """
    img = rectified_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return _encode_jpeg(img)


def _rectify_overlay_encode(
    raw_bytes: bytes,
    camera_matrix: list[list[float]],
    distortion_coefficients: list[float],
    bbox: Bbox,
    region: Optional[Region] = None,
) -> bytes:
    """Sync helper run via asyncio.to_thread — heavy CPU work
    (rawpy decode + cv2.undistort + skimage CLAHE).

    `region` wins when present; `bbox` is the fallback for a payload built by
    an api-worker that predates the polygon.
    """
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(camera_matrix, dtype=float),
        distortion_coefficients=np.array(distortion_coefficients, dtype=float),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics)
    if region:
        return overlay_laser_region_and_encode_jpeg(rectified.data, region)
    return overlay_laser_bbox_and_encode_jpeg(rectified.data, bbox)


def _input_model():
    from fishsense_data_processing_workflow_worker.workflows import (
        preprocess_laser_images_workflow as workflow,
    )

    return workflow.PreprocessLaserImageInput


@activity.defn
async def preprocess_laser_image(payload) -> None:  # type: ignore[no-untyped-def]
    """Download one raw image from the file-exchange, rectify it, draw
    the expected-laser region, and PUT the JPEG back to the
    file-exchange under `{output_folder}/{checksum}.JPG`."""
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    region = payload.region
    activity.logger.info(
        "preprocessing laser image checksum=%s shape=%s",
        payload.checksum,
        "polygon" if region else f"bbox {payload.bbox}",
    )

    client = open_object_store_client()
    raw_bytes = await client.download_raw(payload.checksum)
    jpeg_bytes = await asyncio.to_thread(
        _rectify_overlay_encode,
        raw_bytes,
        payload.camera_matrix,
        payload.distortion_coefficients,
        tuple(payload.bbox),
        region,
    )
    await client.upload_processed_jpeg(
        folder=payload.output_folder,
        checksum=payload.checksum,
        data=jpeg_bytes,
    )

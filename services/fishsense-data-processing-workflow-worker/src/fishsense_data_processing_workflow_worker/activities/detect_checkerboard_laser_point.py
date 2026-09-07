"""Lift one checkerboard frame's laser dot to a 3-D point in camera space.

The per-frame half of checkerboard laser calibration. Download the staged raw
`.ORF`, rectify it, find the board, and intersect the labelled dot's
back-projected ray with the plane the board occupies. The result is exactly
what `_laser_point_in_camera_space` produces from a dive slate — same
`calibration_geometry` seam, same units, same meaning — so the fit that
consumes it is shared.

**Per-image rather than per-dive, and on the CPU queue**, unlike stage 13
which is a single light-queue activity over already-stored labels. This one
holds image bytes: each frame is a rawpy decode peaking at 1-3 GB, which is
the memory ceiling behind the CPU worker's `max_concurrent_activities = 2`.
Splitting it per frame keeps that peak bounded to one frame at a time, makes
each frame independently retryable, and returns a payload of five numbers
rather than an image.

**It rectifies rather than reusing the stage-0.1 JPEG.** That JPEG is the
right frame at the right resolution, and it is tempting. It also has a green
2 px laser-region outline drawn across it and has been through JPEG
quantisation — one of which can sit on the board and break local corner
detection, and both of which move sub-pixel corner positions. Corner positions
are the entire input to the pose, and a slightly wrong pose is a slightly
wrong scale that nothing downstream can see.
"""

from __future__ import annotations

import asyncio

import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
from fishsense_shared import CheckerboardObservation
from temporalio import activity

from fishsense_data_processing_workflow_worker.calibration_geometry import (
    laser_point_on_plane,
    plane_from_correspondences,
)
from fishsense_data_processing_workflow_worker.checkerboard_detection import (
    detect_checkerboard,
)
from fishsense_data_processing_workflow_worker.object_store import (
    open_object_store_client,
)

__all__ = ["detect_checkerboard_laser_point"]


def _observe(
    raw_bytes: bytes,
    payload,
) -> CheckerboardObservation:
    """Sync helper run via `asyncio.to_thread` — rawpy decode, undistort, PnP.

    Returns an observation with `point=None` for any frame that cannot be
    used, rather than raising. A dive is fitted from dozens of frames and only
    needs `MIN_LASER_POINTS` of them, so one unreadable board is ordinary; a
    raise here would fail the whole dive over it.
    """
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(payload.camera_matrix, dtype=float),
        distortion_coefficients=np.array(payload.distortion_coefficients, dtype=float),
        camera_id=None,
    )
    # Rectified, because `plane_from_correspondences` passes zero distortion
    # to solvePnP. Detecting on raw pixels yields a plausible, slightly wrong
    # pose and no error.
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics)

    unusable = CheckerboardObservation(
        image_id=payload.image_id,
        point=None,
        laser_x=payload.laser_x,
        laser_y=payload.laser_y,
    )

    detected = detect_checkerboard(
        rectified.data,
        max_rows=payload.target_rows,
        max_cols=payload.target_cols,
        square_size_m=payload.square_size_m,
    )
    if detected is None:
        return unusable

    plane = plane_from_correspondences(
        detected.body_points,
        detected.image_points,
        intrinsics.camera_matrix,
    )
    if plane is None:
        return unusable

    point = laser_point_on_plane(
        plane,
        np.array([payload.laser_x, payload.laser_y], dtype=float),
        intrinsics.camera_matrix,
    )
    if point is None:
        return unusable

    return CheckerboardObservation(
        image_id=payload.image_id,
        point=[float(point[0]), float(point[1]), float(point[2])],
        laser_x=payload.laser_x,
        laser_y=payload.laser_y,
        detected_rows=detected.rows,
        detected_cols=detected.cols,
    )


def _input_model():
    from fishsense_data_processing_workflow_worker.workflows import (
        perform_checkerboard_calibration_workflow as workflow,
    )

    return workflow.DetectCheckerboardLaserPointInput


@activity.defn
async def detect_checkerboard_laser_point(payload) -> CheckerboardObservation:
    """Where this frame's laser dot sits in space, or None if it can't say."""
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    client = open_object_store_client()
    raw_bytes = await client.download_raw(payload.checksum)
    observation = await asyncio.to_thread(_observe, raw_bytes, payload)

    if observation.point is None:
        activity.logger.info(
            "no usable checkerboard observation image_id=%d checksum=%s",
            payload.image_id,
            payload.checksum,
        )
    else:
        activity.logger.info(
            "checkerboard observation image_id=%d grid=%dx%d depth=%.3fm",
            payload.image_id,
            observation.detected_rows,
            observation.detected_cols,
            observation.point[2],
        )
    return observation

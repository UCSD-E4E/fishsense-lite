"""Activity to compute laser extrinsics for a dive (stage 13).

Ports `scripts/stage13_perform_laser_calibration.ipynb`. The Atanasov
fit delegates to `fishsense_core.laser.calibrate_laser` and the laser
ray projection delegates to `fishsense_core.world_point.WorldPointHandler`
— both validated against pre-refactor prod values within 0.011 deg /
0.39 mm by `scripts/validate_stage13_refactor.py`.

Lives on the data-processing worker (not api-worker) because it pulls
in opencv + fishsense-core for the PnP + laser-fit math; the api-worker
is intentionally kept thin (SDK + Label Studio + scheduling).
"""

from __future__ import annotations

import cv2
import numpy as np
from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive_slate import DiveSlate
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_core.laser import calibrate_laser as _calibrate_laser
from fishsense_core.world_point import WorldPointHandler
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client

INCH_TO_M = 0.0254
MIN_LASER_POINTS = 2

__all__ = ["perform_laser_calibration_activity"]


def _laser_point_in_camera_space(
    label: DiveSlateLabel,
    laser_label: LaserLabel,
    slate: DiveSlate,
    camera_intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    """Lift one slate-laser observation to a 3-D point in camera space.

    Returns None when the observation can't be used (PnP failure, NaN
    ray). Mirrors the per-label kernel from `scripts/validate_stage13_refactor.py`.
    """
    source_points = list(slate.reference_points or [])
    for idx in label.skipped_points or []:
        source_points.pop(idx)
    body_points = np.zeros((len(source_points), 3), dtype=np.float32)
    body_points[:, :2] = (np.array(source_points) / float(slate.dpi)) * INCH_TO_M
    image_space = np.array(label.reference_points)

    ret, rvec, tvec = cv2.solvePnP(
        body_points,
        image_space,
        camera_intrinsics.camera_matrix,
        np.zeros((5,)),
    )
    if not ret:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    camera_space_points = (rotation @ body_points.T + tvec).T
    slate_normal = rotation[:, 2]

    k_inv = np.linalg.inv(camera_intrinsics.camera_matrix)
    laser_image_point = np.array([laser_label.x, laser_label.y])
    ray = WorldPointHandler(k_inv).project_image_point(laser_image_point) * -1
    if np.any(np.isnan(ray)):
        return None

    scale = (slate_normal.T @ camera_space_points[0, :]) / (slate_normal.T @ ray)
    return ray * scale


async def _gather_laser_points(
    fs: Client,
    dive_slate_labels: list[DiveSlateLabel],
    slate: DiveSlate,
    camera_intrinsics: CameraIntrinsics,
) -> list[np.ndarray]:
    laser_points: list[np.ndarray] = []
    for label in dive_slate_labels:
        if label.image_id is None:
            continue
        laser_label = await fs.labels.get_laser_label(image_id=label.image_id)
        if laser_label is None or laser_label.x is None or laser_label.y is None:
            continue
        point = _laser_point_in_camera_space(
            label, laser_label, slate, camera_intrinsics
        )
        if point is not None:
            laser_points.append(point)
    return laser_points


@activity.defn
async def perform_laser_calibration_activity(dive_id: int) -> int | None:
    """Fit laser extrinsics for `dive_id` from its slate-laser labels.

    Returns the persisted `LaserExtrinsics` row id, or None when the dive
    has no `dive_slate_id` / no slate labels (genuine no-op). Raises
    `ValueError` when fewer than `MIN_LASER_POINTS` usable observations
    survive PnP / ray projection — that's a real data problem worth
    surfacing rather than silently producing a degenerate fit.

    Always recomputes; the API endpoint is an upsert. Callers that want
    "skip if already calibrated" should filter on `get_laser_extrinsics`
    before invoking the activity.
    """
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")
        if dive.dive_slate_id is None:
            activity.logger.info(
                "dive_id=%d has no dive_slate_id; skipping calibration", dive_id
            )
            return None

        all_slates = await fs.dive_slates.get() or []
        slate = next((s for s in all_slates if s.id == dive.dive_slate_id), None)
        if slate is None:
            raise ValueError(
                f"dive_id={dive_id}: dive_slate_id={dive.dive_slate_id} not found"
            )
        if slate.dpi is None or not slate.reference_points:
            raise ValueError(
                f"dive_slate_id={slate.id} missing dpi or reference_points"
            )

        dive_slate_labels = await fs.labels.get_dive_slate_labels(dive_id) or []
        if not dive_slate_labels:
            activity.logger.info(
                "dive_id=%d has no dive_slate_labels; skipping calibration",
                dive_id,
            )
            return None

        camera_intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if camera_intrinsics is None:
            raise ValueError(
                f"camera_id={dive.camera_id} has no intrinsics"
            )

        laser_points = await _gather_laser_points(
            fs, dive_slate_labels, slate, camera_intrinsics
        )
        if len(laser_points) < MIN_LASER_POINTS:
            raise ValueError(
                f"dive_id={dive_id}: insufficient laser points "
                f"({len(laser_points)} < {MIN_LASER_POINTS})"
            )

        origin, orientation = _calibrate_laser(
            np.array(laser_points).astype(np.float32)
        )
        # Rust kernel returns origin with z=0 implicit; pad to a 3-vector
        # to match the LaserExtrinsics SDK surface.
        laser_position = np.array(
            [float(origin[0]), float(origin[1]), 0.0], dtype=float
        )
        laser_axis = np.asarray(orientation, dtype=float)

        new_le = LaserExtrinsics(
            laser_position=laser_position,
            laser_axis=laser_axis,
            dive_id=dive_id,
            camera_id=dive.camera_id,
        )
        return await fs.dives.put_laser_extrinsics(dive_id, new_le)

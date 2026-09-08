"""Activity to compute laser extrinsics for a dive (stage 13).

Ports `scripts/stage13_perform_laser_calibration.ipynb`. The Atanasov
fit delegates to `fishsense_core.laser.calibrate_laser`, and the pose +
ray-plane intersection to `calibration_geometry` (which in turn uses
`fishsense_core.world_point.WorldPointHandler`) — both validated against
pre-refactor prod values within 0.011 deg / 0.39 mm by
`scripts/validate_stage13_refactor.py`.

What remains slate-specific here is only how the correspondences are
obtained: template points scaled by the slate's DPI, paired against
hand-placed `DiveSlateLabel.reference_points`. The geometry from there on
is shared with the checkerboard producer described in
`docs/plans/checkerboard-laser-calibration.md`.

Lives on the data-processing worker (not api-worker) because it pulls
in opencv + fishsense-core for the PnP + laser-fit math; the api-worker
is intentionally kept thin (SDK + Label Studio + scheduling).
"""

from __future__ import annotations

import numpy as np
from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_api_sdk.models.dive_slate import DiveSlate
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.laser_extrinsics import LaserExtrinsics
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_core.laser import calibrate_laser as _calibrate_laser
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.utils import get_fs_client
from fishsense_data_processing_workflow_worker.calibration_consistency import (
    check_fit_self_consistency,
)
from fishsense_data_processing_workflow_worker.calibration_geometry import (
    laser_point_on_plane,
    plane_from_correspondences,
)

INCH_TO_M = 0.0254
#: Minimum usable slate-laser observations before a fit is attempted. The
#: api's `MIN_SLATE_LASER_POINTS` must equal this — they are one threshold
#: spelled twice, on opposite sides of the worker boundary, and a dive that
#: clears the cohort's copy but not this one is re-selected hourly forever
#: with nothing written. That is not hypothetical: the cohort used to count
#: completed slate labels rather than observations, and prod dive 347 (18
#: labels, 1 live dot) wedged stage 13 for as long as it was scheduled.
MIN_LASER_POINTS = 2

__all__ = ["perform_laser_calibration_activity"]


def _drop_skipped(points: list, skipped) -> list:
    """Remove `skipped` indices from `points`, resolved against the ORIGINAL list.

    A sequential ``list.pop(idx)`` (the previous implementation, and the
    stage-13 notebook) is only correct for <=1 skipped point: popping index 2
    slides every later element down one, so a subsequent ``pop(5)`` removes what
    was originally index 6 — feeding a wrong 3D<->2D pair into solvePnP. Latent
    while every production label skips <=1 point, but model-assisted labeling
    emits per-point visibility (multi-skip), so it activates the moment that
    ships. Resolve all indices up front, and reject duplicate / out-of-range
    indices. Mirrors `slate_training.contracts.drop_skipped`.
    """
    indices = [int(i) for i in skipped]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate skipped index in {indices}")
    for i in indices:
        if not 0 <= i < len(points):
            raise ValueError(f"skipped index {i} out of range for {len(points)} points")
    drop = set(indices)
    return [p for i, p in enumerate(points) if i not in drop]


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
    source_points = _drop_skipped(
        list(slate.reference_points or []), label.skipped_points or []
    )
    image_points = list(label.reference_points or [])
    # solvePnP pairs body<->image points purely by position, so a count that
    # disagrees with (template - skipped) silently mis-pairs the geometry.
    # Make that a hard error. Mirrors `contracts.template_correspondences`.
    if len(image_points) != len(source_points):
        raise ValueError(
            f"slate label reference_points={len(image_points)} disagrees with "
            f"template {slate.id} ({len(slate.reference_points or [])}) minus "
            f"{len(set(int(i) for i in (label.skipped_points or [])))} skipped "
            f"= {len(source_points)}; refusing to mis-pair solvePnP correspondences"
        )
    # The slate's whole contribution: template points in metres. Everything
    # after `plane_from_correspondences` is target-agnostic and shared with the
    # coming checkerboard producer — see `calibration_geometry`.
    body_points = (np.array(source_points) / float(slate.dpi)) * INCH_TO_M

    plane = plane_from_correspondences(
        body_points,
        np.array(image_points),
        camera_intrinsics.camera_matrix,
    )
    if plane is None:
        return None

    return laser_point_on_plane(
        plane,
        np.array([laser_label.x, laser_label.y]),
        camera_intrinsics.camera_matrix,
    )


async def _gather_laser_points(
    fs: Client,
    dive_slate_labels: list[DiveSlateLabel],
    slate: DiveSlate,
    camera_intrinsics: CameraIntrinsics,
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Lift each usable slate-laser observation to camera space.

    Returns `(laser_points_3d, laser_dots_2d)` in lockstep — the 2D pixels
    feed the post-fit self-consistency gate (the fitted ray must reproject
    onto the very dots it was computed from)."""
    laser_points: list[np.ndarray] = []
    laser_dots: list[tuple[float, float]] = []
    for label in dive_slate_labels:
        if label.image_id is None:
            activity.heartbeat()
            continue
        laser_label = await fs.labels.get_laser_label(image_id=label.image_id)
        if laser_label is None or laser_label.x is None or laser_label.y is None:
            activity.heartbeat()
            continue
        point = _laser_point_in_camera_space(
            label, laser_label, slate, camera_intrinsics
        )
        if point is not None:
            laser_points.append(point)
            laser_dots.append((float(laser_label.x), float(laser_label.y)))
        activity.heartbeat()
    return laser_points, laser_dots


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
            raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

        laser_points, laser_dots = await _gather_laser_points(
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

        # Self-consistency gate: the fitted ray must reproject onto the 2D
        # dots it was computed from. Raises (no persist) when it doesn't —
        # mixed dot populations (specular-reflection mislabels, prod dive 77)
        # or corrupt slate poses otherwise ship a calibration whose length
        # errors reach +137% downstream.
        check_fit_self_consistency(
            laser_position,
            laser_axis,
            camera_intrinsics.camera_matrix,
            np.array(laser_dots, dtype=float),
        )

        new_le = LaserExtrinsics(
            laser_position=laser_position,
            laser_axis=laser_axis,
            dive_id=dive_id,
            camera_id=dive.camera_id,
        )
        return await fs.dives.put_laser_extrinsics(dive_id, new_le)

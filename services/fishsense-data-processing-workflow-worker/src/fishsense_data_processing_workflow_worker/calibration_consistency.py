"""Stage-13 self-consistency gate: a laser calibration must reproject onto
the 2D laser dots it was fit from.

Motivated by prod dive 77 (2026-08-04): labelers clicked the laser's specular
reflection on ~half the pool slate frames, so the dots formed two parallel
lines ~45 px apart. The 3D fit split the difference — 14 deg off the dive's
own dot line, origin far outside the fleet family — and the persisted
calibration, borrowed by a fish-model dive, produced +31..+137% length errors.

The invariant is cheap and assumption-free: the fitted 3D ray, projected
through the camera matrix, must coincide with the total-least-squares line
through the input dots. Fleet baseline (every good calibration, all cameras):
<=1.6 deg / <=4 px median; the dive-77 failure was 14 deg / 17 px median.
Reprojection error also *predicts* measurement bias (0.9 px borrow -> +-3%
lengths; 2-4 px -> -4..-15%), so the thresholds below bound downstream
accuracy, not just fit hygiene.

Pure numpy — no cv2 — so it is unit-testable everywhere.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CalibrationInconsistentError",
    "DEFAULT_MAX_ANGLE_DEG",
    "DEFAULT_MAX_MEDIAN_OFFSET_PX",
    "MIN_DOT_SPAN_PX",
    "check_fit_self_consistency",
]

# Fleet-derived thresholds: good fits sit at <=1.6 deg / <=4 px median, the
# known-bad fit at 14 deg / 17 px. The gap is wide; these sit in the middle
# with margin for noisier-but-sane dives.
DEFAULT_MAX_ANGLE_DEG = 3.0
DEFAULT_MAX_MEDIAN_OFFSET_PX = 8.0

# Below this dot spread the 2D line direction is numerically meaningless
# (e.g. MIN_LASER_POINTS=2 nearly-coincident observations), so the gate
# abstains rather than rejecting on noise.
MIN_DOT_SPAN_PX = 50.0


class CalibrationInconsistentError(ValueError):
    """The fitted laser ray does not reproject onto the dots it came from."""


def _tls_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares 2D line through `points` -> (centroid, unit dir)."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    return centroid, vt[0]


def check_fit_self_consistency(
    laser_position: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
    dots_xy: np.ndarray,
    *,
    max_angle_deg: float = DEFAULT_MAX_ANGLE_DEG,
    max_median_offset_px: float = DEFAULT_MAX_MEDIAN_OFFSET_PX,
    min_span_px: float = MIN_DOT_SPAN_PX,
) -> None:
    # pylint: disable=too-many-locals
    # Flat geometric pipeline (project ray -> fit both lines -> two
    # scalars); splitting it would smear one computation across helpers.
    """Raise `CalibrationInconsistentError` when the fitted ray's projection
    disagrees with the 2D laser dots it was computed from.

    `dots_xy` is an (N, 2) array of the laser pixels whose observations fed
    the fit. Abstains (returns) when N < 2 or the dots' spread is below
    `min_span_px` — too degenerate to define a comparison line.
    """
    dots = np.asarray(dots_xy, dtype=float)
    if dots.ndim != 2 or dots.shape[0] < 2:
        return
    span = float(np.linalg.norm(dots.max(axis=0) - dots.min(axis=0)))
    if span < min_span_px:
        return

    origin = np.asarray(laser_position, dtype=float)
    axis = np.asarray(laser_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.asarray(camera_matrix, dtype=float)

    # Project the fitted ray across the plausible working depth range.
    depths = np.linspace(0.2, 4.0, 200)
    ts = (depths - origin[2]) / axis[2]
    ray_points = origin[None, :] + ts[:, None] * axis[None, :]
    homogeneous = (k @ ray_points.T).T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]

    proj_centroid, proj_dir = _tls_line(projected)
    _, dot_dir = _tls_line(dots)

    cos_angle = min(1.0, abs(float(proj_dir @ dot_dir)))
    angle_deg = float(np.degrees(np.arccos(cos_angle)))

    normal = np.array([-proj_dir[1], proj_dir[0]])
    median_offset_px = float(np.median(np.abs((dots - proj_centroid) @ normal)))

    if angle_deg > max_angle_deg:
        raise CalibrationInconsistentError(
            f"fitted laser ray reprojects at angle {angle_deg:.2f} deg to the "
            f"input dot line (gate {max_angle_deg} deg); median offset "
            f"{median_offset_px:.1f}px. The fit disagrees with its own "
            f"observations — mixed dot populations (e.g. specular-reflection "
            f"mislabels) or corrupt slate poses. Refusing to persist."
        )
    if median_offset_px > max_median_offset_px:
        raise CalibrationInconsistentError(
            f"fitted laser ray reprojects with median offset "
            f"{median_offset_px:.1f}px from the input dot line "
            f"(gate {max_median_offset_px}px); angle {angle_deg:.2f} deg. "
            f"Refusing to persist."
        )

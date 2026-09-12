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
from fishsense_shared.calibration_bounds import MAX_BASELINE_M, MIN_BASELINE_M

__all__ = [
    "CalibrationImplausibleError",
    "CalibrationInconsistentError",
    "DEFAULT_MAX_ANGLE_DEG",
    "DEFAULT_MAX_BASELINE_M",
    "DEFAULT_MAX_MEDIAN_OFFSET_PX",
    "DEFAULT_MIN_BASELINE_M",
    "MIN_DOT_SPAN_PX",
    "check_baseline_plausible",
    "check_fit_self_consistency",
]

# --- baseline plausibility -------------------------------------------------
#
# `check_fit_self_consistency` below cannot see the baseline, and that is
# structural rather than a threshold being too loose. It compares the fitted
# ray's PROJECTION against the 2D dots. Many different 3D rays project to the
# same image line -- the whole family lying in the plane through the camera
# centre and that line -- and sliding the laser's offset along that family
# moves where the ray crosses z=0 while leaving the projection identical. So
# it constrains two of the ray's four degrees of freedom, and the baseline
# sits in the two it cannot. Same epipolar blindness already documented for
# reprojection residual and scale.
#
# The baseline is checkable anyway, because it is not a property of the dive:
# it is the physical offset between camera and laser on one rig. Measured
# 2026-09-11 over all 35 stored calibrations, its interquartile range is
# 9.99-10.45 cm -- half a centimetre, across both producers, every camera and
# two years of dives.
#
# Against that, 8 fits sat outside 8-13 cm: 2.35, 2.60, 4.81, 4.90, 6.25,
# 6.91, 16.01 and 22.22 cm. They back 12 dives and 663 of 3,104 measurements,
# and grade -75% to +45% against the known-length targets. **Two are
# slate-derived**, which is why this lives here beside the shared gate and not
# in the checkerboard path.

#: Re-exported from `fishsense_shared.calibration_bounds`, which owns them.
#:
#: The api needs the identical numbers — it excludes an already-stored
#: implausible calibration from counting as a calibration at all, so the dive
#: re-enters the calibration cohorts instead of being measured against a fit we
#: know is wrong. Two copies would be the drift this repo keeps rediscovering,
#: and here disagreement is especially quiet: the api would hand a dive back
#: for recalibration that this module then persists unchanged, hourly, forever.
DEFAULT_MIN_BASELINE_M = MIN_BASELINE_M
DEFAULT_MAX_BASELINE_M = MAX_BASELINE_M


class CalibrationImplausibleError(ValueError):
    """The fitted laser sits at a physically implausible offset from the camera."""


def check_baseline_plausible(
    laser_position,
    *,
    min_baseline_m: float = DEFAULT_MIN_BASELINE_M,
    max_baseline_m: float = DEFAULT_MAX_BASELINE_M,
) -> None:
    """Raise `CalibrationImplausibleError` when the fitted baseline is not a rig.

    `laser_position` is the fit's origin: where the laser ray crosses the
    camera's z=0 plane. Its z component is padding (both producers emit the
    x/y pair and set z to 0), so only the in-plane offset is read -- taking a
    three-component norm would make the answer depend on a convention that
    carries no information.

    **Checked before persisting, not after.** A calibration that reaches
    `LaserExtrinsics` is immediately borrowable by sibling dives through
    `calibration_dive_id`, so a bad one stops being one dive's problem the
    moment it is written.

    **A refusal wedges the dive, and that is deliberate but not free.** Both
    calibration cohorts select on "has no `LaserExtrinsics` row", so a dive
    refused here stays eligible and is re-selected every hour, re-staging its
    raw `.ORF`s from the NAS each time and — because the selectors are
    `ORDER BY id LIMIT 1` — blocking every higher-id dive behind it. That is
    the same head-of-line shape as prod dive 347. It is the right trade against
    silently wrong lengths, but it means the refusal has to tell an operator
    what to do, which is why the message names the remedies.
    """
    position = np.asarray(laser_position, dtype=float)
    baseline = float(np.linalg.norm(position[:2]))

    # `isfinite` explicitly and first: every comparison against NaN is False,
    # so a bare range test would ACCEPT a NaN baseline. The safe reading of
    # "no measurable offset" is refusal.
    if not np.isfinite(baseline) or not min_baseline_m <= baseline <= max_baseline_m:
        raise CalibrationImplausibleError(
            f"fitted laser baseline {baseline * 100:.2f} cm is outside the "
            f"plausible range {min_baseline_m * 100:.1f}-"
            f"{max_baseline_m * 100:.1f} cm. The baseline is a property of the "
            f"camera+laser rig, not of the dive, and the fleet sits at "
            f"~10.4 cm; a value this far off means the fit, not the hardware. "
            f"It would scale every depth and every length by that factor, "
            f"invisibly to reprojection residual and to the self-consistency "
            f"gate. Refusing to persist. This dive stays in the calibration "
            f"cohort and will be re-selected hourly, blocking higher-id dives: "
            f"either fix its observations, or park it with Priority.NONE and a "
            f"note, or (checkerboard dives) clear its calibration target via "
            f"DELETE /api/v1/dives/{{id}}/calibration-target/."
        )


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

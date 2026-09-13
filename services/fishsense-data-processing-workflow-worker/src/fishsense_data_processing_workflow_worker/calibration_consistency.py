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
    "CalibrationDoesNotDescribeDiveError",
    "CalibrationUnderdeterminedError",
    "DEFAULT_MAX_DIVE_MEDIAN_OFFSET_PX",
    "DEFAULT_MAX_DIVE_P90_OFFSET_PX",
    "MIN_DIVE_DOTS",
    "MIN_OBSERVATION_LEVER_M",
    "MIN_DOT_SPAN_PX",
    "check_baseline_plausible",
    "check_calibration_describes_dive",
    "check_observation_geometry",
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


def _project_ray_to_image_line(
    laser_position: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The image of the fitted laser ray -> (centroid, unit dir).

    Both projection gates need this same line: one compares it against the
    dots the fit came from, the other against every dot in the dive. Sampling
    the ray across the plausible working depth range and fitting is exact, not
    an approximation — for a pinhole camera the image of a straight 3-D line
    is a straight line.
    """
    origin = np.asarray(laser_position, dtype=float)
    axis = np.asarray(laser_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    depths = np.linspace(0.2, 4.0, 200)
    ts = (depths - origin[2]) / axis[2]
    ray_points = origin[None, :] + ts[:, None] * axis[None, :]
    homogeneous = (np.asarray(camera_matrix, dtype=float) @ ray_points.T).T
    return _tls_line(homogeneous[:, :2] / homogeneous[:, 2:3])


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

    proj_centroid, proj_dir = _project_ray_to_image_line(
        laser_position, laser_axis, camera_matrix
    )
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


# --- does the calibration describe the dive it will measure? ---------------
#
# The gate above compares the fitted ray against the dots it was computed
# from, so it is satisfied by construction, and it abstains when those dots
# are too few -- which is exactly the case it most needs to catch. Prod dive
# 347 got through both: the fit came from ONE frame carrying a duplicate laser
# label at the identical pixel, so `dots_xy` had N=1 and the check returned
# silently, while `MIN_LASER_POINTS = 2` was satisfied by counting the
# duplicate. The resulting calibration sits 9.4 px from the dive's own 321
# laser dots.
#
# This gate asks the other question, and it is not tautological: project the
# fitted ray and compare it against EVERY live dot in the dive, including the
# measurement frames the fit never saw. Those dots are where the lengths come
# from, so agreement there is the thing that actually matters, and
# disagreement means the laser was not in the same state for the frames being
# measured as for the frames it was calibrated from -- what a mid-dive re-seat
# does (prod dive 490: board burst and fish frames 0.82 deg apart).
#
# Measured over all 32 stored calibrations against their dives' live dots
# (2026-09-13), median perpendicular offset in px:
#
#     0.43-0.88   twenty dives
#     1.31-1.55   six dives
#     2.01, 2.27  dives 279, 509
#     3.33, 3.92  dives 341, 349   <- real burst-vs-frames disagreement
#     5.37        dive 498         <- also caught by the baseline gate
#     9.42        dive 347         <- the degenerate fit, caught by nothing else
#
# and p90 reaches 6.07 on the healthy side (dive 341) against 18.44 (347) and
# 29.69 (498). Both bounds sit between the populations rather than against the
# healthy one, the same rule the baseline gate uses and for the same reason: a
# refusal wedges the dive in its cohort, so a false positive is expensive.
#
# **Blind to the baseline, unavoidably.** Sliding the laser's offset along the
# family of rays sharing an image line leaves the projection identical, so five
# of the six known-bad baselines sit at 0.43-1.35 px here and pass. That is
# what `check_baseline_plausible` is for. Neither gate subsumes the other.
DEFAULT_MAX_DIVE_MEDIAN_OFFSET_PX = 6.0
DEFAULT_MAX_DIVE_P90_OFFSET_PX = 12.0
#: Below this many usable dots the dive cannot answer the question, so the
#: gate abstains rather than refusing. The degenerate-observation case is
#: caught by the observation count in the calibration activity, not here.
MIN_DIVE_DOTS = 6


class CalibrationDoesNotDescribeDiveError(ValueError):
    """The fitted ray disagrees with the dive's own laser dots."""


def check_calibration_describes_dive(
    laser_position: np.ndarray,
    laser_axis: np.ndarray,
    camera_matrix: np.ndarray,
    dive_dots_xy: np.ndarray,
    *,
    max_median_offset_px: float = DEFAULT_MAX_DIVE_MEDIAN_OFFSET_PX,
    max_p90_offset_px: float = DEFAULT_MAX_DIVE_P90_OFFSET_PX,
    min_dots: int = MIN_DIVE_DOTS,
    min_span_px: float = MIN_DOT_SPAN_PX,
) -> None:
    """Raise `CalibrationDoesNotDescribeDiveError` when the fitted ray's
    projection disagrees with the laser dots of the dive it will measure.

    `dive_dots_xy` is an (N, 2) array of every live laser pixel in the dive,
    not just the observations that fed the fit. Non-finite rows are dropped.
    Abstains when fewer than `min_dots` usable dots remain or their spread is
    below `min_span_px`: a dive that cannot define a line cannot answer this.
    """
    dots = np.asarray(dive_dots_xy, dtype=float)
    if dots.ndim != 2 or dots.shape[1] != 2:
        return
    dots = dots[np.isfinite(dots).all(axis=1)]
    if dots.shape[0] < min_dots:
        return
    span = float(np.linalg.norm(dots.max(axis=0) - dots.min(axis=0)))
    if span < min_span_px:
        return

    centroid, direction = _project_ray_to_image_line(
        laser_position, laser_axis, camera_matrix
    )
    normal = np.array([-direction[1], direction[0]])
    offsets = np.abs((dots - centroid) @ normal)
    median_px = float(np.median(offsets))
    p90_px = float(np.percentile(offsets, 90))

    if median_px > max_median_offset_px:
        raise CalibrationDoesNotDescribeDiveError(
            f"calibration does not describe its dive: the fitted ray projects "
            f"with median offset {median_px:.1f}px from the dive's own "
            f"{dots.shape[0]} laser dots (gate {max_median_offset_px}px, p90 "
            f"{p90_px:.1f}px). The laser was not in the same state for the "
            f"frames being measured as for the frames it was calibrated from, "
            f"or the fit came from too few observations to constrain it. "
            f"Refusing to persist."
        )
    if p90_px > max_p90_offset_px:
        raise CalibrationDoesNotDescribeDiveError(
            f"calibration does not describe its dive: p90 offset {p90_px:.1f}px "
            f"from the dive's own {dots.shape[0]} laser dots (gate "
            f"{max_p90_offset_px}px, median {median_px:.1f}px). A subset of the "
            f"dive disagrees with the fit. Refusing to persist."
        )


# --- is the fit determined by its observations at all? ---------------------
#
# `MIN_LASER_POINTS` in the calibration activity counts observations; it does
# not ask where they are. What determines the fitted ray's DIRECTION is the
# lever arm -- the spread of the observations along the ray -- against the
# label noise. One pixel of dot-label noise at range z is z/f metres of
# lateral error, so it rotates the fit by about (z/f)/lever radians, and the
# measured sensitivity of length to that angle is ~30 % per degree at 2 m. So
# two observations a metre apart determine the direction far better than
# sixteen at one range, and the count is the wrong quantity to bound.
#
# Measured over the 32 stored calibrations whose observations are recoverable
# (2026-09-13), lever arm and the resulting % of length per px of label noise:
#
#     dive 341   30 obs   1.55 m   0.1 %
#     dive 471    2 obs   1.56 m   0.7 %
#     dive 279    3 obs   1.47 m   0.6 %
#     dive 465    3 obs   1.34 m   0.6 %
#     dive 383    2 obs   1.09 m   0.7 %   <- tightest sound calibration
#     dive 349    2 obs   0.26 m   5.8 %   <- refused
#     dive 107   16 obs   0.06 m   5.5 %   <- refused
#     dive 347    1 obs   0.00 m   degenerate
#
# Dive 107 is why this gate is needed: sixteen observations, a dot span wide
# enough that `check_fit_self_consistency` does not abstain, and a 12.95 cm
# baseline that `check_baseline_plausible` accepts as the fleet's high
# extreme -- with 6 cm of lever. Dive 526 had the same single-distance
# geometry and was caught only because its fit happened to collapse to
# 2.00 cm. The bound sits between the populations, nearer the bad side
# because a refusal wedges the dive in its cohort.
MIN_OBSERVATION_LEVER_M = 0.60


class CalibrationUnderdeterminedError(ValueError):
    """The observations do not span enough range to determine the ray."""


def check_observation_geometry(
    observations_xyz: np.ndarray,
    *,
    min_lever_m: float = MIN_OBSERVATION_LEVER_M,
) -> None:
    """Raise `CalibrationUnderdeterminedError` when the 3-D observations are
    too tightly grouped along the ray to fix its direction.

    `observations_xyz` is the (N, 3) array of camera-space laser points that
    will be fitted. Non-finite rows are dropped before measuring. Unlike the
    other gates this one does not abstain on small N: one observation, or
    several at one distance, is exactly the case it exists to refuse.
    """
    points = np.asarray(observations_xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CalibrationUnderdeterminedError(
            "laser observations are not an (N, 3) array of camera-space points"
        )
    points = points[np.isfinite(points).all(axis=1)]
    n = points.shape[0]
    lever = 0.0 if n < 2 else float(points[:, 2].max() - points[:, 2].min())
    if lever < min_lever_m:
        raise CalibrationUnderdeterminedError(
            f"laser observations span only {lever:.2f} m of range "
            f"({n} observation{'s' if n != 1 else ''}; gate {min_lever_m} m). "
            f"The lever arm, not the count, fixes the fitted direction: at this "
            f"spread one pixel of dot-label noise moves every length by several "
            f"percent. Shoot the target at two clearly different distances. "
            f"Refusing to persist."
        )

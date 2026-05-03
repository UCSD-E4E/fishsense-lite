"""Per-dive RANSAC line fit on positive laser labels.

VENDORED — DO NOT EDIT IN ISOLATION. Source of truth:
    https://github.com/UCSD-E4E/2026-05-02_laser_detector
    src/laser_detector/preprocessing/line_fit.py @ 3d5d2e8 (2026-05-02)

The laser-detector repo is in early development ("nothing trained yet").
We duplicated this module instead of taking a git dep so prod isn't
coupled to a moving research repo. Replace the duplicate with a real
import once the laser-detector publishes a stable release; see the
"Open follow-ups" section of the repo-root CLAUDE.md.

Two surface differences from the upstream module:

* Upstream operates on polars DataFrames as part of its batch pipeline;
  the public API here takes a numpy array of (x, y) positives plus a
  thin convenience that returns outlier indices, so the caller (an
  activity that already has a ``LaserLabel`` list from the SDK) doesn't
  pull polars in.
* The polars helpers ``fit_lines_per_dive`` and ``flag_label_outliers``
  are dropped — the per-dive RANSAC / outlier kernel is the only piece
  the validation activity needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Minimum positive labels required to attempt a line fit. A degenerate fit
# (2 points) always succeeds; we want enough redundancy for RANSAC to mean
# something.
MIN_POINTS_FOR_LINE = 5

# RANSAC inlier tolerance in pixels (perpendicular distance). 4K frames + a
# 3 px laser blob → ~4 px is a generous-but-not-loose tolerance for label
# noise. Tunable; logged so we can revisit.
RANSAC_INLIER_TOL_PX = 4.0

# Max RANSAC iterations.
RANSAC_MAX_ITERS = 200

# Confidence threshold below which we say the line is ambiguous and the prior
# should not be applied. Eigenvalue ratio (along-line spread / perp spread).
LINE_CONFIDENCE_THRESHOLD = 5.0

# MAD → consistent estimator of σ for normally distributed residuals.
MAD_TO_SIGMA = 1.4826

# Floor on the MAD-derived σ used by ``flag_outliers``. On very small dives
# whose RANSAC inliers happen to be sub-pixel-tight, MAD collapses to ~0 and
# every label gets flagged. Labels can't reasonably be more precise than ~1 px
# at native 4K resolution, so any threshold below this is non-physical.
LABEL_NOISE_MAD_FLOOR_PX = 1.0

# Default σ multiple used by ``flag_outliers``. 3σ leaves a ~0.3% false-flag
# rate under Gaussian noise, which on a typical-size dive (~100 positives) is
# well under one expected false flag.
DEFAULT_OUTLIER_SIGMA = 3.0


@dataclass
class LineFit:  # pylint: disable=too-many-instance-attributes
    """A normalized line ``a*x + b*y + c = 0`` plus quality metrics."""

    a: float
    b: float
    c: float
    n_points: int
    inlier_count: int
    inlier_fraction: float
    residual_std: float  # perp-distance std among inliers, in px (RANSAC tightness)
    label_noise_mad: float  # 1.4826 * MAD over ALL positive labels' perp distance, in px
    line_confidence: float  # along-line spread / perp spread (covariance eigenratio)

    @property
    def is_confident(self) -> bool:
        return self.line_confidence >= LINE_CONFIDENCE_THRESHOLD

    def perpendicular_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perpendicular distance from each (x,y) to this line, in pixels."""
        return np.abs(self.a * x + self.b * y + self.c)


def _fit_line_total_least_squares(xy: np.ndarray) -> tuple[float, float, float]:
    """Fit a 2D line via SVD on centered points (total least squares).

    Returns normalized ``(a, b, c)`` for ``a*x + b*y + c = 0``.
    """
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    # SVD: smallest singular vector is the normal to the best-fit line
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    a, b = float(normal[0]), float(normal[1])
    norm = float(np.hypot(a, b))
    if norm == 0.0:
        return 1.0, 0.0, 0.0
    a, b = a / norm, b / norm
    c = float(-(a * centroid[0] + b * centroid[1]))
    return a, b, c


def _ransac_line(
    xy: np.ndarray,
    tol_px: float,
    max_iters: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    """RANSAC line fit. Returns ``(a, b, c, inlier_mask)``."""
    n = xy.shape[0]
    best_inliers: np.ndarray | None = None
    best_count = -1

    for _ in range(max_iters):
        idx = rng.choice(n, size=2, replace=False)
        p0, p1 = xy[idx[0]], xy[idx[1]]
        dx, dy = p1 - p0
        norm = float(np.hypot(dx, dy))
        if norm == 0.0:
            continue
        a, b = -dy / norm, dx / norm
        c = -(a * p0[0] + b * p0[1])
        dist = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
        inliers = dist < tol_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 2:
        a, b, c = _fit_line_total_least_squares(xy)
        return a, b, c, np.ones(n, dtype=bool)

    a, b, c = _fit_line_total_least_squares(xy[best_inliers])
    dist = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
    inliers = dist < tol_px
    return a, b, c, inliers


def _line_confidence(xy: np.ndarray, a: float, b: float) -> float:
    """Ratio of along-line variance to perpendicular variance.

    A high ratio means the points are spread out along the line (well-determined
    direction). A low ratio means they cluster, leaving the line direction
    ambiguous.
    """
    centered = xy - xy.mean(axis=0)
    along = np.array([-b, a])
    perp = np.array([a, b])
    var_along = float(np.var(centered @ along))
    var_perp = float(np.var(centered @ perp))
    if var_perp <= 1e-9:
        return float("inf")
    return var_along / var_perp


def fit_dive_line(
    xy: np.ndarray,
    *,
    tol_px: float = RANSAC_INLIER_TOL_PX,
    max_iters: int = RANSAC_MAX_ITERS,
    rng: np.random.Generator | None = None,
) -> LineFit | None:
    """Fit a line to one dive's positive labels.

    Returns ``None`` when fewer than ``MIN_POINTS_FOR_LINE`` positives are
    available — RANSAC on 2-4 points is degenerate.
    """
    if xy.shape[0] < MIN_POINTS_FOR_LINE:
        return None
    rng = rng or np.random.default_rng(0)
    a, b, c, inliers = _ransac_line(xy, tol_px=tol_px, max_iters=max_iters, rng=rng)
    inlier_xy = xy[inliers]
    n = xy.shape[0]
    inlier_count = int(inliers.sum())

    dist_inliers = np.abs(a * inlier_xy[:, 0] + b * inlier_xy[:, 1] + c)
    residual_std = float(np.std(dist_inliers))
    confidence = _line_confidence(inlier_xy, a, b)

    # MAD on ALL positive labels — the population the outlier flag will be
    # applied against. ``residual_std`` is bounded by the RANSAC tolerance and
    # so under-states true label-noise scale; MAD is robust to the gross
    # outliers in the tail.
    dist_all = np.abs(a * xy[:, 0] + b * xy[:, 1] + c)
    label_noise_mad = float(
        MAD_TO_SIGMA * np.median(np.abs(dist_all - np.median(dist_all)))
    )

    return LineFit(
        a=a,
        b=b,
        c=c,
        n_points=n,
        inlier_count=inlier_count,
        inlier_fraction=inlier_count / n,
        residual_std=residual_std,
        label_noise_mad=label_noise_mad,
        line_confidence=confidence,
    )


def flag_outliers(
    xy: np.ndarray,
    fit: LineFit,
    *,
    sigma: float = DEFAULT_OUTLIER_SIGMA,
    mad_floor_px: float = LABEL_NOISE_MAD_FLOOR_PX,
) -> np.ndarray:
    """Boolean mask marking labels whose perpendicular distance to ``fit``
    exceeds ``sigma * max(label_noise_mad, mad_floor_px)``.

    Returns an all-False mask when ``fit`` is not confident — callers
    should not act on outlier flags from a low-confidence line.

    The MAD floor handles small-N dives where MAD collapses to sub-pixel
    values and would otherwise flag every label.
    """
    if not fit.is_confident:
        return np.zeros(xy.shape[0], dtype=bool)
    effective_mad = max(fit.label_noise_mad, mad_floor_px)
    threshold = sigma * effective_mad
    perp = fit.perpendicular_distance(xy[:, 0], xy[:, 1])
    return perp > threshold

"""The stage-0.1 expected-laser region has to cover every rig, not just one.

Stage 0.1 runs *before* the dive has a `LaserExtrinsics` row of its own —
calibration is stage 13 — so `LASER_REGION_POLYGON` is a single constant
applied to every dive, and the only way to know it is right is to measure the
rig population it has to cover. This module pins that measurement so the
constant can't drift back to a guess.

Two independent legs, because neither is sufficient alone:

* `PROD_CALIBRATIONS` is the geometry. Projecting each fitted laser ray across
  the working depth range says where the dot *can* be. It is exact but it is
  a sample of 13 — only 13 of the 262 dives carrying laser labels have ever
  been calibrated.
* `PROD_DIVE_LASER_LOCI` is the observation, and it is what caught the bug.
  The uncalibrated rigs sit measurably further right than the calibrated 13:
  dives 253/397/455/468 carry their per-dive *median* laser x at 2213..2267,
  past the 3-sigma envelope of the calibrated set. A region derived from the
  calibrations alone would clip them.

Corpus measured 2026-08-27 against prod: 13 `LaserExtrinsics` rows and the
31,322 completed non-superseded `LaserLabel` rows spanning 262 dives.
"""

from __future__ import annotations

import numpy as np
import pytest

from fishsense_shared.laser_region import (
    DEFAULT_LASER_BBOX,
    LASER_REGION_POLYGON,
    WORKING_DEPTH_RANGE_M,
    point_in_laser_region,
)

# (laser_origin_xy, laser_axis, fx, fy, cx, cy) for all 13 prod calibrations.
# The origin's z is 0.0 in every prod row -- `perform_laser_calibration_activity`
# pads the Rust kernel's 2-vector origin with an implicit zero -- so it is
# reconstructed in `_project_ray` rather than repeated 13 times.
PROD_CALIBRATIONS = [
    ((-0.030078, -0.099278), (0.014452, 0.038441, 0.999156), 2833.28, 2858.77, 2009.76, 1407.99),
    ((-0.032060, -0.099302), (0.001308, 0.027590, 0.999618), 2832.89, 2857.55, 2027.31, 1492.53),
    ((-0.031704, -0.099556), (0.026947, 0.029967, 0.999188), 2827.30, 2852.24, 1993.26, 1443.37),
    ((-0.031174, -0.100156), (0.031329, 0.047707, 0.998370), 2855.29, 2881.07, 2031.79, 1447.69),
    ((-0.030574, -0.099056), (0.005685, 0.032688, 0.999449), 2832.66, 2855.88, 1968.03, 1452.81),
    ((-0.029878, -0.100020), (0.022015, 0.027924, 0.999368), 2825.15, 2850.61, 2061.85, 1458.56),
    ((-0.029473, -0.099408), (0.029015, 0.013736, 0.999485), 2841.37, 2863.76, 2004.36, 1476.25),
    ((-0.029347, -0.098302), (0.001327, 0.026194, 0.999656), 2833.28, 2858.77, 2009.76, 1407.99),
    ((-0.031076, -0.097454), (0.002612, -0.015737, 0.999873), 2832.66, 2855.88, 1968.03, 1452.81),
    ((-0.031321, -0.113729), (-0.017094, 0.068653, 0.997494), 2825.15, 2850.61, 2061.85, 1458.56),
    ((-0.032651, -0.100247), (0.022480, 0.032909, 0.999206), 2832.89, 2857.55, 2027.31, 1492.53),
    ((-0.030791, -0.097424), (0.008205, -0.003560, 0.999960), 2827.30, 2852.24, 1993.26, 1443.37),
    ((-0.030248, -0.099551), (0.034429, 0.085700, 0.995726), 2832.89, 2857.55, 2027.31, 1492.53),
]

# The dives whose observed laser population defines an edge of the region:
# (dive_id, x_p5, y_p5, x_p95, y_p95) over that dive's completed non-superseded
# labels. p5/p95 rather than min/max because a few percent of every dive's
# labels are specular-reflection mislabels that survived the RANSAC supersede
# pass -- e.g. (2217, 2088) recurs verbatim across dives 246/375/446, a fixed
# artifact of the rig rather than a laser anyone shot.
PROD_DIVE_LASER_LOCI = [
    (424, 1792, 1216, 1945, 1380),  # leftmost observed rig
    (262, 1795, 1064, 1872, 1308),
    (253, 2202, 1397, 2328, 1570),  # rightmost -- an uncalibrated rig
    (397, 2196, 1158, 2268, 1364),  # ditto
    (468, 2205, 1431, 2247, 1540),  # ditto
    (437, 2041, 543, 2303, 1425),  # highest; the V-Slate 7 dive
    (446, 1965, 1313, 2110, 1763),  # lowest
]

# Widest per-dive median over all 262 dives -- a median cannot be a mislabel,
# so these are the hardest floor the box has to clear.
PROD_DIVE_MEDIAN_EXTREMES = {"x_min": 1768, "x_max": 2367, "y_min": 1069, "y_max": 1616}

_FRAME_W, _FRAME_H = 4014, 3016


def _polygon_area(poly) -> float:
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _project_ray(calibration, depths: np.ndarray) -> np.ndarray:
    """Pixels swept by one rig's laser ray across `depths`."""
    origin_xy, axis, fx, fy, cx, cy = calibration
    origin = np.array([origin_xy[0], origin_xy[1], 0.0], dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    t = (depths - origin[2]) / axis[2]
    points = origin[None, :] + t[:, None] * axis[None, :]
    return np.column_stack(
        [
            fx * points[:, 0] / points[:, 2] + cx,
            fy * points[:, 1] / points[:, 2] + cy,
        ]
    )


def _contains(xs, ys) -> bool:
    """Every (x, y) inside the region, via the predicate we actually ship.

    Deliberately not a second implementation: a test-local copy would let the
    shipped one drift while the corpus below kept passing.
    """
    xs = np.asarray(xs, float).ravel()
    ys = np.asarray(ys, float).ravel()
    return all(point_in_laser_region(float(x), float(y)) for x, y in zip(xs, ys))


@pytest.mark.parametrize("calibration", PROD_CALIBRATIONS)
def test_region_contains_every_calibrated_laser_ray(calibration):
    """Each fitted ray, projected across the working depth range, lands
    inside the region."""
    depths = np.linspace(*WORKING_DEPTH_RANGE_M, 500)
    uv = _project_ray(calibration, depths)
    assert _contains(uv[:, 0], uv[:, 1]), (
        f"laser ray projects to x {uv[:, 0].min():.0f}..{uv[:, 0].max():.0f} "
        f"y {uv[:, 1].min():.0f}..{uv[:, 1].max():.0f}, outside the region"
    )


@pytest.mark.parametrize("locus", PROD_DIVE_LASER_LOCI)
def test_region_contains_every_observed_dive_locus(locus):
    """Each edge-defining dive's observed laser population is inside the region.

    All four corners of the dive's locus, not just the diagonal: the region is
    not axis-aligned, so a rectangle can poke out of it at a corner while both
    of its diagonal ends sit inside.
    """
    dive_id, x1, y1, x2, y2 = locus
    xs = np.array([x1, x1, x2, x2])
    ys = np.array([y1, y2, y1, y2])
    assert _contains(
        xs, ys
    ), f"dive {dive_id} lasers span x {x1}..{x2} y {y1}..{y2}, outside the region"


def test_region_clears_every_per_dive_median_with_margin():
    """No dive's *typical* laser may sit on or outside an edge.

    The pre-2026-08-27 rectangle failed exactly here: at (1800, 700, 2400, 1600) two
    dives had their median laser outside it entirely (32 px left of the left
    edge, 16 px below the bottom edge), so the box was pointing labelers away
    from the laser for whole dives rather than at it.
    """
    x1, y1, x2, y2 = DEFAULT_LASER_BBOX
    e = PROD_DIVE_MEDIAN_EXTREMES
    margins = {
        "left": e["x_min"] - x1,
        "right": x2 - e["x_max"],
        "top": e["y_min"] - y1,
        "bottom": y2 - e["y_max"],
    }
    assert min(margins.values()) >= 100, f"insufficient margin: {margins}"


def test_region_is_a_hint_not_the_whole_frame():
    """A region that covers everything guides nobody. Guards the obvious wrong
    fix for a clipped laser -- widening until the misses stop."""
    x1, y1, x2, y2 = DEFAULT_LASER_BBOX
    assert 0 <= x1 < x2 <= _FRAME_W and 0 <= y1 < y2 <= _FRAME_H
    assert _polygon_area(LASER_REGION_POLYGON) / (_FRAME_W * _FRAME_H) < 0.15


def test_region_is_convex_and_wound_consistently():
    """`_contains` above is a convexity test, and the data-worker draws the
    vertices in the order given -- a concave or self-crossing ring would make
    both silently wrong rather than fail."""
    poly = np.asarray(LASER_REGION_POLYGON, float)
    assert len(poly) >= 4
    edge = np.roll(poly, -1, axis=0) - poly
    nxt = np.roll(edge, -1, axis=0)
    cross = edge[:, 0] * nxt[:, 1] - edge[:, 1] * nxt[:, 0]
    assert np.all(cross > 0) or np.all(cross < 0), "region is not convex"


def test_bbox_is_the_regions_bounding_box():
    """`bbox` is the rolling-deploy fallback: a data-worker that predates the
    polygon draws it instead, so it has to be a superset of the region, and
    the tightest one available. See `PreprocessLaserImagesInput`."""
    poly = np.asarray(LASER_REGION_POLYGON, int)
    assert DEFAULT_LASER_BBOX == [
        int(poly[:, 0].min()),
        int(poly[:, 1].min()),
        int(poly[:, 0].max()),
        int(poly[:, 1].max()),
    ]


# --- the predicate itself ---------------------------------------------------
#
# `point_in_laser_region` is what the laser detector gates on, so a bug here
# either silently disables the gate (everything accepted) or throws away every
# prediction. Neither shows up as an error anywhere downstream: a rejected
# prediction is indistinguishable from "the model found no laser".


def test_centre_of_the_region_is_inside():
    cx = sum(v[0] for v in LASER_REGION_POLYGON) / len(LASER_REGION_POLYGON)
    cy = sum(v[1] for v in LASER_REGION_POLYGON) / len(LASER_REGION_POLYGON)
    assert point_in_laser_region(cx, cy)


@pytest.mark.parametrize("vertex", LASER_REGION_POLYGON)
def test_vertices_are_inside(vertex):
    """Edges and corners count as inside. The region is a decision boundary
    fitted with 150 px of margin, so half-open semantics would be arbitrary --
    and a point exactly on an edge is where a ray-casting test gets it wrong."""
    assert point_in_laser_region(float(vertex[0]), float(vertex[1]))


def test_edge_midpoints_are_inside():
    for i, start in enumerate(LASER_REGION_POLYGON):
        end = LASER_REGION_POLYGON[(i + 1) % len(LASER_REGION_POLYGON)]
        assert point_in_laser_region(
            (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ), f"edge {i} midpoint rejected"


@pytest.mark.parametrize(
    "x,y,why",
    [
        (0, 0, "frame origin"),
        (4013, 3015, "opposite frame corner"),
        (1600, 1800, "inside the bbox, but in the corner the polygon cuts"),
        (2450, 1750, "ditto, bottom-right"),
        (1590, 1300, "just left of the region's left edge"),
        (1580, 1905, "bbox corner the polygon cuts off (bottom-left)"),
        (2470, 395, "bbox corner the polygon cuts off (top-right)"),
        (2217, 2088, "the recurring specular-reflection artifact"),
    ],
)
def test_points_outside_are_rejected(x, y, why):
    assert not point_in_laser_region(float(x), float(y)), why


def test_the_cut_corners_are_what_a_rectangle_would_have_wrongly_accepted():
    """The whole reason this is a polygon: all four bbox corners are outside
    it, so a bbox gate would accept predictions the geometry rules out."""
    x1, y1, x2, y2 = DEFAULT_LASER_BBOX
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    assert not any(point_in_laser_region(float(x), float(y)) for x, y in corners)


def test_explicit_region_argument_overrides_the_default():
    """The region crosses the wire, so the caller's copy wins -- a data-worker
    must gate on the polygon the api-worker sent, not one baked into its own
    build."""
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_laser_region(5, 5, square)
    assert not point_in_laser_region(50, 50, square)


@pytest.mark.parametrize("degenerate", [[], [[0, 0]], [[0, 0], [1, 1]]])
def test_a_degenerate_region_admits_nothing(degenerate):
    """Fails closed. An empty region reaching here means the contract lost the
    polygon; accepting everything would disable the gate with no signal."""
    assert not point_in_laser_region(5, 5, degenerate)


def test_winding_order_does_not_matter():
    """The constant is authored clockwise today; reversing it must not invert
    the test."""
    reversed_poly = list(reversed(LASER_REGION_POLYGON))
    cx = sum(v[0] for v in LASER_REGION_POLYGON) / len(LASER_REGION_POLYGON)
    cy = sum(v[1] for v in LASER_REGION_POLYGON) / len(LASER_REGION_POLYGON)
    assert point_in_laser_region(cx, cy, reversed_poly)
    assert not point_in_laser_region(0, 0, reversed_poly)

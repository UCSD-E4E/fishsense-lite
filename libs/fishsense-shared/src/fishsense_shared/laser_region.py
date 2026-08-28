"""Where the laser dot can be, and the test for whether a point is in it.

The laser is mounted ~3 cm left of and ~10 cm below the camera, so its dot
does not sit in a fixed place: it traces an epipolar streak that runs up and to
the left as the subject gets closer, converging on a per-rig asymptote as it
gets further away. This region is the union of those streaks over every rig.

It lives here, beside `preprocess_contracts` and `object_store`, for the same
reason those do: it is an agreement *between* the two workers, so neither owns
it. Stage 0.1 sends it to the data-worker to draw as the labeling hint, and the
laser-detector stage sends it to reject predictions that fall outside it. It
started out inline in the stage-0.1 resolver, and was moved when the second
consumer appeared rather than copied -- a copied constant is exactly the shape
of drift `duplicate-code` cannot see, since the two would be different names
for the same numbers.

Measured 2026-08-27 against prod (13 `LaserExtrinsics` rows; 31,322 completed
non-superseded `LaserLabel` rows across 262 dives): the convex hull of the
projected calibration rays plus every dive's observed laser locus, dilated by
150 px and simplified to 8 vertices. It holds every calibrated ray, all 190
well-populated dives' loci, and 99.90% of the labels -- each of the ~30
stragglers being a specular-reflection mislabel that survived the RANSAC
supersede pass, not a laser. `(2217, 2088)` recurs verbatim across dives
246/375/446, which is a fixed artifact of the rig rather than anything shot.

It is a polygon rather than a rectangle because that union is genuinely not
axis-aligned, and the corners a rectangle adds are where the laser cannot be.
At equal coverage the polygon is 26% smaller than the tightest rectangle
(1.12 vs 1.52 Mpx). A *rotated* rectangle was measured and rejected: it comes
out 3.5 degrees off axis and saves 4%, because the per-rig streak is diagonal
while the rig-to-rig spread is horizontal and fills the box back in.

The predecessor, [1800, 700, 2400, 1600], was the original notebook constant.
Across the corpus it clipped the left edge (103 labels), the bottom (76) and
the top (33), and two dives had their *median* laser outside it altogether.

Deriving it from the 13 calibrations alone would still have clipped: only 13 of
the 262 labelled dives have ever been calibrated, and the uncalibrated rigs sit
measurably further right -- dives 253/397/455/468 carry median laser x at
2213..2267, past the 3-sigma envelope of the calibrated set.

All coordinates are **rectified** pixels -- the space labelers place
`LaserLabel.x/y` and the space `LaserDetector.predict(rectify_output=True)`
returns. The detector's own `rig_prior_bbox` is a different thing in a
different frame (sensor coordinates, and much looser); do not conflate them.

`tests/test_laser_region.py` pins both derivation legs against the prod corpus.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

__all__ = [
    "DEFAULT_LASER_BBOX",
    "LASER_REGION_POLYGON",
    "WORKING_DEPTH_RANGE_M",
    "point_in_laser_region",
]

# Subject distances the rig actually works at, padded either side of the
# observed range. `LaserDepth` over 1109 prod images spans 0.44-5.45 m
# (p01 0.54, p50 1.71, p99 3.50).
WORKING_DEPTH_RANGE_M: Tuple[float, float] = (0.35, 8.0)

# Convex, in draw order. See the module docstring for how it was measured.
LASER_REGION_POLYGON: List[List[int]] = [
    [1580, 570],
    [1700, 465],
    [2335, 395],
    [2455, 525],
    [2470, 1610],
    [2185, 1890],
    [1920, 1905],
    [1625, 1365],
]

# The polygon's bounding box, and only ever that -- `test_laser_region` pins
# the two together. It exists because the api-worker and the data-worker deploy
# independently (in-slot converge vs. `kubectl apply` on NRP, often days
# apart), so a data-worker that predates the polygon has to keep drawing
# something correct. See `PreprocessLaserImagesInput.laser_region`.
DEFAULT_LASER_BBOX: List[int] = [1580, 395, 2470, 1905]


def point_in_laser_region(
    x: float,
    y: float,
    region: Sequence[Sequence[float]] | None = None,
) -> bool:
    """Whether `(x, y)` lies inside the (convex) laser region, edges included.

    Convex, so a point is inside iff it is on the same side of every directed
    edge -- no ray casting, no winding number, and no special case for a point
    sitting exactly on an edge, which a ray-casting test decides by rounding.
    Written without numpy so the data-worker can call it inside a workflow
    sandbox and the api-worker can call it without importing an array library.

    A degenerate region (fewer than 3 vertices) admits nothing rather than
    everything: this gates whether a model prediction is believed, and failing
    open would silently disable the gate.
    """
    poly = [tuple(v) for v in (region if region is not None else LASER_REGION_POLYGON)]
    if len(poly) < 3:
        return False

    positive = negative = False
    for index, (ax, ay) in enumerate(poly):
        bx, by = poly[(index + 1) % len(poly)]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross > 0:
            positive = True
        elif cross < 0:
            negative = True
        if positive and negative:
            return False
    return True

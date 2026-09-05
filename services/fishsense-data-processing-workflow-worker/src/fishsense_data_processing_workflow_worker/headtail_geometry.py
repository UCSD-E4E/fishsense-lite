"""Geometry for the head/tail predict stage — no model, no GPU, no I/O.

Kept separate from the activity because these are the parts that fail
*plausibly*. A wrong crop origin displaces every keypoint by a constant and
still produces a fish-shaped answer; a wrong silhouette ratio quietly changes
which predictions get seeded. Both are invisible in a spot check and obvious in
a unit test, so they live where a unit test can reach them.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = ["crop_origin", "lift_point", "mask_at_point", "silhouette_ratio"]


def crop_origin(
    laser_x: float,
    laser_y: float,
    frame_w: int,
    frame_h: int,
    crop_w: int,
    crop_h: int,
) -> Tuple[int, int]:
    """Top-left of the crop window centred on the laser dot, clamped inside
    the frame.

    Clamped rather than padded or truncated: a padded window shows the model a
    black band and a truncated one changes its aspect, and both would apply
    exactly to the frames where the fish is already near an edge — the ones
    most likely to be cut. Clamping moves the window instead, which keeps the
    fish in view and the input shape constant.
    """
    max_x = max(0, frame_w - crop_w)
    max_y = max(0, frame_h - crop_h)
    ox = int(min(max(0, laser_x - crop_w // 2), max_x))
    oy = int(min(max(0, laser_y - crop_h // 2), max_y))
    return ox, oy


def lift_point(
    point: Sequence[float], origin_x: int, origin_y: int
) -> Tuple[float, float]:
    """Move a crop-local point back into rectified-frame coordinates.

    `HeadTailPrediction` is defined in rectified-frame pixels — the space
    `LaserLabel.x/y` and the labeler's own clicks live in — so this has to
    happen before anything is persisted.
    """
    return (float(point[0]) + origin_x, float(point[1]) + origin_y)


def mask_at_point(
    masks: Iterable[np.ndarray], points: Sequence[Sequence[float]]
) -> Optional[np.ndarray]:
    """The laser gate: the first mask whose pixel at a laser dot is set.

    "Which fish" is a lookup, not a search — that is what the dot buys, and on
    the measured corpus it picked a different fish than "largest detection"
    would have on 20% of predictions. Every laser point is tried because an
    image may carry more than one valid label (461 prod images do), and
    first-hit-wins is the behaviour that was measured.

    Returns None when no dot lands on any mask, which is an abstention, not an
    error.
    """
    masks = list(masks)
    for px, py in points:
        xi, yi = int(round(px)), int(round(py))
        for mask in masks:
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                return mask
    return None


def silhouette_ratio(mask_area_px: int, length_px: float) -> Optional[float]:
    """`mask_area / length**2` — how fish-shaped a detection is.

    A real fish silhouette runs ~0.15-0.30. Recorded on every row and applied
    as a seed-time band rather than a predict-time filter, so it can be retuned
    from data already collected without re-predicting anything.

    None for a degenerate length: a zero-length prediction has no shape, and
    None says that more honestly than an infinity would.
    """
    if not length_px:
        return None
    return mask_area_px / (length_px * length_px)

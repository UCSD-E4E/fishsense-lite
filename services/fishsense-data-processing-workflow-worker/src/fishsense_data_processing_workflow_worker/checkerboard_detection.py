"""Find a checkerboard's interior corners and pair them with board coordinates.

This is the checkerboard half of what a `DiveSlate` supplies to stage 13:
`(body_points, image_points)` for `calibration_geometry`. Everything after
that — the pose, the plane, the ray-plane intersection, the Atanasov fit, the
self-consistency gate, `LaserExtrinsics` — is already shared and already knows
nothing about what the target was. See
`docs/plans/checkerboard-laser-calibration.md`.

**The detector is asked for a small grid, and told the real one.**
`findChessboardCornersSB` with `CALIB_CB_LARGER` returns whatever grid it
actually found, and reports its shape in the metadata; asking for the nominal
board instead is worse in both directions, measured against OpenCV 4.13:

* a partial view fails outright. A board with its right third out of frame
  returns nothing for the exact `patternSize` and a clean 10 x 10 sub-grid for
  a small one — and occlusion, glare and the frame edge cut the board down
  often enough that the survey saw sub-grids on many frames.
* a wrong-but-plausible `patternSize` also fails outright, so an operator who
  linked the wrong target would see "no board here" rather than a diagnosis.

Reading the shape back also *is* the safety property. A nominal full-board
grid paired against a partial detection mis-pairs every correspondence, and
`solvePnP` does not refuse that: it returns success and a confident,
meaningless pose. There is no residual check anywhere between here and
`LaserExtrinsics`, so nothing downstream would notice.

**Orientation is free, and deliberately unconstrained.** The detected grid
comes back transposed or mirrored depending on how the board sits in frame
(the survey saw the same board reported as both 10 x 14 and 14 x 10). It does
not matter: the board is a symmetric grid, so its indexing is only defined up
to the symmetries of a rectangle, a mirrored assignment is realised by a real
rotation (the board flipped about an in-plane axis) which merely flips the
plane normal, and the ray-plane intersection divides one normal by the other
so the sign cancels. Only the plane reaches the answer — never the in-plane
pose — which is why the classic checkerboard headaches are all irrelevant
here. `test_checkerboard_detection.py` pins that.

**Detect on the RECTIFIED image, at full resolution.** `solvePnP` is called
with zero distortion, which is only correct because the pipeline feeds it
undistorted imagery; raw pixels give a plausible, slightly wrong pose and no
error. Downscaling is its own hazard — at half scale the 2023 board came back
one column short, and a silently smaller grid is a mis-pairing risk, not just
fewer points.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["DetectedCheckerboard", "MIN_GRID_SIDE", "detect_checkerboard"]


#: Smallest grid side accepted, in interior corners.
#:
#: A one- or two-corner-wide strip is near-collinear and does not determine a
#: plane — and `solvePnP` will not say so. Six coincident points return
#: *success* and a confident, meaningless pose, so an automatic producer has
#: to gate on its own detection quality because nothing after it will. Three
#: is the smallest grid that spans two independent in-plane directions with a
#: point to spare in each; every sub-grid the corpus survey returned was
#: comfortably above it (the smallest was 6 x 14).
MIN_GRID_SIDE = 3

#: The grid asked for. See the module docstring — the real one is read back
#: from the detector's metadata, so this only has to be small enough not to
#: exclude a heavily-cropped view.
_REQUESTED_PATTERN = (MIN_GRID_SIDE, MIN_GRID_SIDE)

#: `EXHAUSTIVE` raises the detection rate, `ACCURACY` upsamples for sub-pixel
#: precision (these corners are the whole input to the pose), and `LARGER`
#: is what lets the detector return a grid other than the one requested.
_DETECTION_FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_LARGER


@dataclass(frozen=True)
class DetectedCheckerboard:
    """A detected grid and its correspondences, index-aligned.

    `body_points` are board coordinates in **metres**, `image_points` the
    rectified pixels they landed on, both `(rows * cols, 2)` in the detector's
    own row-major order. Ordering is the contract `solvePnP` depends on: it
    pairs them purely by position.
    """

    rows: int
    cols: int
    body_points: np.ndarray
    image_points: np.ndarray


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _fits_declared_board(rows: int, cols: int, max_rows: int, max_cols: int) -> bool:
    """Whether a `rows x cols` detection can be part of the declared board.

    Compared as unordered pairs because the detected grid's orientation is
    free (see the module docstring), so a 10 x 14 detection of a 14 x 10 board
    is the same board.

    A detection *larger* than the declared board is refused rather than
    trimmed: it means the frame holds a different board, so the stored pitch
    does not apply, and a wrong pitch is a wrong scale — the one error term
    the reprojection residual provably cannot see. The 2025 pool-test board
    (24 x 17) against the E4E board (14 x 10) is exactly that case.

    **Elementwise, on both sides.** Comparing the sorted pairs with `<=`
    directly is a lexicographic list compare, not a "fits inside" test: it
    accepts 9 x 20 against a 10 x 14 board, because 9 < 10 settles the
    comparison and the 20 is never looked at. A partial view of the 2025 board
    lands in exactly that shape and would then be fitted at the E4E board's
    pitch.
    """
    short, long = sorted((rows, cols))
    max_short, max_long = sorted((max_rows, max_cols))
    return short <= max_short and long <= max_long


def detect_checkerboard(
    image: np.ndarray,
    *,
    max_rows: int,
    max_cols: int,
    square_size_m: float,
) -> DetectedCheckerboard | None:
    """Detect the board in `image` and pair its corners with board coordinates.

    `image` must be the **rectified** frame, full resolution, grayscale or
    BGR. `max_rows` / `max_cols` are the declared board's interior corners and
    act as an upper bound, not a target. `square_size_m` is the measured grid
    pitch.

    Returns None when no usable board is found — no detection, a degenerate
    strip, a grid larger than the declared board, or metadata that disagrees
    with the corners returned. Every one of those is a case where continuing
    would hand `solvePnP` a correspondence set it would accept and answer
    wrongly, so the caller drops the frame and the dive is fitted from the
    others.
    """
    found, corners, meta = cv2.findChessboardCornersSBWithMeta(
        _to_grayscale(image), _REQUESTED_PATTERN, _DETECTION_FLAGS
    )
    if not found or corners is None or meta is None:
        return None

    rows, cols = int(meta.shape[0]), int(meta.shape[1])
    if min(rows, cols) < MIN_GRID_SIDE:
        return None
    if not _fits_declared_board(rows, cols, max_rows, max_cols):
        return None

    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    # Belt and braces on the detector's own contract: if the metadata grid and
    # the corner count ever disagree, the reshape below would pair the wrong
    # points and solvePnP would accept it silently.
    if len(image_points) != rows * cols:
        return None

    # Board coordinates on the DETECTED grid, at the measured pitch. Its
    # origin is wherever the detected sub-grid starts and its axes are
    # whichever way round the detector indexed the board — both are in-plane
    # pose, which nothing downstream reads.
    col_index, row_index = np.meshgrid(np.arange(cols), np.arange(rows))
    body_points = np.stack([col_index.ravel(), row_index.ravel()], axis=1).astype(
        np.float64
    ) * float(square_size_m)

    return DetectedCheckerboard(
        rows=rows,
        cols=cols,
        body_points=body_points,
        image_points=image_points,
    )

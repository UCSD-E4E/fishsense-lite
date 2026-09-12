"""Audit a dive's recorded laser depths against the checkerboard's solvePnP depths.

WHAT THE BOARD IS AND IS NOT INDEPENDENT OF. `solvePnP` on the board's corners
at a known 42 mm pitch gives an absolute distance per frame without consulting
the laser extrinsics. Whether that is an *independent* check of the laser
depth depends entirely on which frames are compared, and this script does two
different things on two populations:

  * Frames where the laser dot lies ON the board (`dot_on_board=True`). These
    are the frames the calibration was fitted from -- the ray-plane
    intersection that produced the fit is the same one this script recomputes
    -- so agreement here is near-tautological. On every audited dive,
    including 490, it agrees to +-0.6 %. Read it as a self-consistency /
    degeneracy check of the fit on its own observations, never as evidence
    that the calibration is right.
  * Frames where a rigid target is in view beside the board and the dot is on
    the TARGET (`--model`). The board's depth is then a genuinely independent
    range for the target, and `head-tail pixels x depth` is constant for a
    rigid object. That is what convicted dive 490: over seven such frames the
    product against BOARD depth held at 873-883 (+-0.6 %) across a 2.75x range
    while the product against RECORDED depth drifted 747 -> 519. It is a real
    scale check, but it needs the board and the target in the same frame,
    which is rare (7 frames on 490) and absent on ordinary dives.

For the general case -- any rigid object, no board in frame -- use the
scale-free range-trend audit in `audit_length_range_trend.py` /
`range_trend.py`: a rigid object must read the same length at every range,
so the slope of length against depth exposes an in-plane calibration error
without a board or a known length.

WHAT DIVE 490 TURNED OUT TO BE (2026-09-12, read-only prod timeline). The
folder `ED-00/FSL-02D/LaserCalibration3` holds 68 Weasly Fish frames shot
19:00:49-19:02 and a checkerboard burst shot 19:07:25-19:08:53. The preceding
folder, `LaserCalibration2` (dive 489), is a board burst ending 19:00:40 --
nine seconds before 490's fish frames begin. The two fitted calibrations
differ by 0.82 deg IN-PLANE and 0.06 deg out-of-plane, with baselines 10.33
and 10.24 cm. Re-measuring 490's fish frames under 489's row: median -1.7 %,
p90 +1.3 %, residual in-plane angle 0.02 deg, and the 23-point range trend is
gone. So the laser moved between 19:02 and 19:07 -- AFTER the fish frames and
BEFORE the board burst that calibrated them. The earlier reading here, that
the fish frames were shot after the laser had moved, had the order reversed;
the fish frames belong to the calibration that precedes them. Dives 491 and
492 (18:33-18:39, before either burst) fit neither row -- 0.21 deg from 490's
and 0.65 deg from 489's -- and their contemporaneous burst, LaserCalibration1
(dive 488), is parked for caustics with its board labels lost.

WHY IT IS A SCRIPT AND NOT A GATE. Two candidate pipeline checks were built
against this failure and both were falsified on the corpus, which is worth
recording so they are not rebuilt:

  * Comparing a borrowed calibration's laser line against its source's. Over
    the 20 active borrow pairs it separated cleanly, but per-model validation
    showed the flagged dives were no less accurate than the unflagged ones
    (median |deviation| 2.2% vs 2.4%) and dive 490 was not flagged at all,
    because it measures against its own calibration. The offset it measures is
    PERPENDICULAR, and depth is set by position ALONG the line.
  * Gating a new fit against its own camera's baseline history, judged by
    the median length error. The three baseline outliers graded within 1.6 %
    on the median (498 at 9.51 cm, 502 at 8.90, 107 at 12.95), which read as
    "an 8% baseline error buys a 1% length error". That reading was wrong:
    the range trend later showed 498 and 502's borrowers at -14 to -18 % up
    close rising to ~0 at 4 m, a flat scale error cancelling against an angle
    error at the median. The median is not a calibration check; the range
    trend is, and the baseline floor moved from 7.8 to 9.7 cm on its evidence.

And the dot-on-board check is tautological on its own domain, as above, while
only about a third of frames detect a board at all. So this is an instrument
you run when you want to trust an accuracy number, not something to put in the
hourly path.

USAGE
    uv run --package fishsense-data-processing-workflow-worker python \\
        services/fishsense-data-processing-workflow-worker/scripts/\\
audit_scale_against_checkerboard.py 490 492 --model "Weasly Fish"

Read the output as: `px*board` constant across a wide depth span means the
board ranged the target consistently. If `px*recorded` then drifts, the recorded
depths are wrong -- that is the dive-490 signature. Both constant but at
different values means a scale offset, not a pointing error.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from fishsense_api_sdk.client import Client

from fishsense_data_processing_workflow_worker.calibration_geometry import (
    laser_point_on_plane,
    plane_from_correspondences,
)
from fishsense_data_processing_workflow_worker.config import settings
from fishsense_data_processing_workflow_worker.object_store import (
    open_object_store_client,
)

#: The E4E board: 14x10 INTERIOR corners (a 15x11 grid of squares) at a 42 mm
#: pitch. Both numbers are the `calibrationtarget` row, not guesses -- and its
#: notes are worth reading before trusting a result to better than ~1%: the
#: pitch was read with a millimetre rule across one square, so it carries about
#: +-1.2% of scale, in the one direction reprojection residual cannot see.
#:
#: DO NOT back-solve the pitch from the fish models. They are the validation
#: set; a pitch fitted to them would make every accuracy number
#: self-confirming.
BOARD_COLS, BOARD_ROWS = 14, 10
BOARD_PITCH_M = 0.042

#: Where stage 0.1 writes the rectified JPEG the laser labels were made on.
#: Detection MUST happen on the rectified image: `plane_from_correspondences`
#: passes zero distortion to solvePnP for exactly that reason, and raw pixels
#: there give a plausible, slightly wrong pose and no error.
LASER_JPEG_FOLDER = "preprocess_jpeg"


@dataclass
class FrameAudit:
    """One frame's two independent depth estimates, plus the scale product."""

    image_id: int
    recorded_depth_m: float
    board_depth_m: float
    dot_on_board: bool
    headtail_px: float | None

    @property
    def deviation_pct(self) -> float:
        return 100.0 * (self.recorded_depth_m - self.board_depth_m) / self.board_depth_m

    @property
    def px_times_board(self) -> float | None:
        if self.headtail_px is None:
            return None
        return self.headtail_px * self.board_depth_m

    @property
    def px_times_recorded(self) -> float | None:
        if self.headtail_px is None:
            return None
        return self.headtail_px * self.recorded_depth_m


def _board_body_points() -> np.ndarray:
    """The board's corners in its own frame, metres, z=0.

    Order matters and is not cosmetic: `solvePnP` pairs body to image points
    purely by position. `findChessboardCorners` returns them row-major with the
    column index varying fastest, so `meshgrid(...).ravel()` matches it exactly.
    """
    cols, rows = np.meshgrid(np.arange(BOARD_COLS), np.arange(BOARD_ROWS))
    body = np.zeros((BOARD_ROWS * BOARD_COLS, 3), dtype=np.float64)
    body[:, 0] = cols.ravel() * BOARD_PITCH_M
    body[:, 1] = rows.ravel() * BOARD_PITCH_M
    return body


def detect_board(gray: np.ndarray) -> np.ndarray | None:
    """Sub-pixel interior corners, or None when no board is fully in view.

    A partially-occluded board fails outright rather than returning a subset,
    which is why coverage runs near a third: the target frequently runs off the
    frame edge in these captures.
    """
    found, corners = cv2.findChessboardCorners(
        gray,
        (BOARD_COLS, BOARD_ROWS),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return corners.reshape(-1, 2)


def board_depth_at_dot(
    corners: np.ndarray,
    dot: np.ndarray,
    camera_matrix: np.ndarray,
) -> float | None:
    """Depth of the board's plane along the dot's ray, in metres.

    Reuses `calibration_geometry` rather than transcribing the ray-plane
    intersection, which is the drift that module was extracted to prevent.
    """
    plane = plane_from_correspondences(_board_body_points(), corners, camera_matrix)
    if plane is None:
        return None
    point = laser_point_on_plane(plane, dot, camera_matrix)
    if point is None:
        return None
    depth = float(point[2])
    return depth if depth > 0 else None


def dot_lies_on_board(corners: np.ndarray, dot: np.ndarray) -> bool:
    """Is the dot inside the detected board?

    Load-bearing. The intersection happily extends the plane, so a dot on a
    nearer object returns the plane's depth in that direction and reads as a
    spurious disagreement. Only an on-board dot makes the depth comparison a
    like-for-like test; an off-board dot is still usable for the `px*board`
    scale product, which needs the board only as a ranging reference.
    """
    hull = cv2.convexHull(corners.astype(np.float32))
    return cv2.pointPolygonTest(hull, (float(dot[0]), float(dot[1])), False) >= 0


async def _headtail_span_px(fs: Client, image_id: int) -> float | None:
    """Pixel distance between the head and tail marks, or None if unlabelled."""
    headtail = await fs.labels.get_headtail_label(image_id=image_id)
    if headtail is None:
        return None
    corners = (headtail.head_x, headtail.head_y, headtail.tail_x, headtail.tail_y)
    if any(c is None for c in corners):
        return None
    return float(
        np.hypot(headtail.head_x - headtail.tail_x, headtail.head_y - headtail.tail_y)
    )


async def audit_dive(
    fs: Client,
    store,
    dive_id: int,
    model: str | None,
) -> list[FrameAudit]:
    """Audit every frame of `dive_id` that has a dot, a depth and a board."""
    dive = await fs.dives.get(dive_id)
    if dive is None or dive.camera_id is None:
        print(f"dive {dive_id}: no dive or no camera", file=sys.stderr)
        return []
    intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
    if intrinsics is None:
        print(f"dive {dive_id}: camera {dive.camera_id} has no intrinsics",
              file=sys.stderr)
        return []
    camera_matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)

    depths = {d.image_id: d for d in await fs.images.get_laser_depths(dive_id)}
    audits: list[FrameAudit] = []
    for image_id, depth_row in sorted(depths.items()):
        if depth_row.depth_m is None or depth_row.depth_m <= 0:
            continue
        laser = await fs.labels.get_laser_label(image_id=image_id)
        if laser is None or laser.x is None or laser.y is None:
            continue
        image = await fs.images.get(image_id=image_id)
        if image is None or image.checksum is None:
            continue
        try:
            blob = await store.download_processed_jpeg(
                LASER_JPEG_FOLDER, image.checksum
            )
        # pylint: disable-next=broad-exception-caught
        except Exception as exc:  # an unreadable JPEG skips the frame, not the run
            print(f"  image {image_id}: no jpeg ({type(exc).__name__})",
                  file=sys.stderr)
            continue
        gray = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE)
        corners = detect_board(gray)
        if corners is None:
            continue
        dot = np.array([float(laser.x), float(laser.y)])
        board_depth = board_depth_at_dot(corners, dot, camera_matrix)
        if board_depth is None:
            continue

        headtail_px = None
        if model is not None:
            headtail_px = await _headtail_span_px(fs, image_id)

        audits.append(
            FrameAudit(
                image_id=image_id,
                recorded_depth_m=float(depth_row.depth_m),
                board_depth_m=board_depth,
                dot_on_board=dot_lies_on_board(corners, dot),
                headtail_px=headtail_px,
            )
        )
    return audits


def report(dive_id: int, audits: list[FrameAudit]) -> None:
    """Print the per-frame table and the two constancy statistics."""
    if not audits:
        print(f"\ndive {dive_id}: no auditable frames "
              f"(a board must be fully in view)")
        return

    print(f"\n=== dive {dive_id}: {len(audits)} auditable frames ===")
    print(f"{'image':>8} {'rec_z':>8} {'board_z':>8} {'dev':>8} "
          f"{'on_board':>9} {'ht_px':>8} {'px*board':>9} {'px*rec':>8}")
    for a in sorted(audits, key=lambda x: x.board_depth_m):
        pb = "" if a.px_times_board is None else f"{a.px_times_board:9.1f}"
        pr = "" if a.px_times_recorded is None else f"{a.px_times_recorded:8.1f}"
        ht = "" if a.headtail_px is None else f"{a.headtail_px:8.1f}"
        print(f"{a.image_id:>8} {a.recorded_depth_m:>8.3f} {a.board_depth_m:>8.3f} "
              f"{a.deviation_pct:>+7.1f}% {str(a.dot_on_board):>9} {ht:>8} "
              f"{pb:>9} {pr:>8}")

    on_board = [a for a in audits if a.dot_on_board]
    if on_board:
        dev = np.array([a.deviation_pct for a in on_board])
        print(f"  dot-on-board depth check: n={len(dev)} median "
              f"{np.median(dev):+.1f}%  range {dev.min():+.1f}% to {dev.max():+.1f}%")
        print("    (near-tautological -- these are the frames the fit came from)")

    scaled = [a for a in audits if a.px_times_board is not None]
    if len(scaled) >= 3:
        pb = np.array([a.px_times_board for a in scaled])
        pr = np.array([a.px_times_recorded for a in scaled])
        zs = np.array([a.board_depth_m for a in scaled])
        print(f"  scale product over {len(scaled)} frames, board depth "
              f"{zs.min():.2f}-{zs.max():.2f} m:")
        print(f"    px*board    median {np.median(pb):8.1f}  "
              f"spread {100.0 * pb.std() / np.mean(pb):5.1f}%")
        print(f"    px*recorded median {np.median(pr):8.1f}  "
              f"spread {100.0 * pr.std() / np.mean(pr):5.1f}%")
        print("    A rigid target holds px*depth CONSTANT. If px*board is flat "
              "and px*recorded is not,")
        print("    the recorded depths are wrong -- the dive-490 signature.")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dive_ids", type=int, nargs="+")
    parser.add_argument(
        "--model",
        default=None,
        help="also compute the scale product, which needs a rigid target of "
             "fixed length in frame (e.g. 'Weasly Fish'). Without it only the "
             "dot-on-board depth check runs.",
    )
    args = parser.parse_args()

    store = open_object_store_client()
    async with Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    ) as fs:
        for dive_id in args.dive_ids:
            report(dive_id, await audit_dive(fs, store, dive_id, args.model))


if __name__ == "__main__":
    asyncio.run(main())

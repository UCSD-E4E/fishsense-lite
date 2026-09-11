"""Render one calibration frame's detected lattice, for human verification.

The per-frame half of checkerboard lattice verification. Download the staged
raw `.ORF`, rectify it, find the board, draw the lattice the detector found,
and upload the result as a JPEG a labeler will judge.

**Why a human is in this loop at all.** A uniformly mis-latticed detection —
every corner two squares apart while `body_points` labels them one apart — is
still a perfect grid, hence an exact homography of the modelled grid, so
`grid_residual_px` reads ~0 and admits it. It also clears
`_fits_declared_board` (a coarse lattice is *smaller* than the declared board)
and `check_fit_self_consistency` (which compares 2-D dots, and a pure scale
error does not move them). The board is then placed at a fraction of its true
distance and every depth, the fitted baseline and every length scales with it.
Measured 2026-09-11: five of fourteen checkerboard calibrations carry a
baseline that is a simple multiple of the consensus 10.4 cm. Nothing in the
pipeline can confirm or kill that. A person looking at the lattice can.

**It mirrors `detect_checkerboard_laser_point`'s admission filter exactly** —
`detect_checkerboard`, then `board_hull` + `point_in_laser_region` — and that
is the point rather than an accident. The study asks what the *fit* consumed,
so its population has to be the fit's population; a frame the calibration path
rejected is not evidence about the calibration and would only pad the labeler's
queue. Both call the same two public helpers in the same order rather than
sharing a private one, because the two stages diverge immediately afterwards
(this one draws and uploads, that one intersects a ray and returns five
numbers) and a shared wrapper would have to carry both futures.

**It rectifies rather than reusing the stage-0.1 JPEG**, for the same reason
the calibration path does: that JPEG carries a drawn laser-region outline and
has been through JPEG quantisation, either of which can sit on the board and
move a corner. Here it matters twice over — the labeler is judging corner
positions, so the frame they see must be the frame the detector saw.
"""

from __future__ import annotations

import asyncio

import numpy as np
from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage
from fishsense_shared import CheckerboardLatticeRender
from fishsense_shared.laser_region import point_in_laser_region
from fishsense_shared.object_store import CHECKERBOARD_LATTICE_JPEG_FOLDER
from temporalio import activity

from fishsense_data_processing_workflow_worker.activities.preprocess_headtail_image import (  # noqa: E501  pylint: disable=line-too-long
    encode_rectified_jpeg,
)
from fishsense_data_processing_workflow_worker.checkerboard_detection import (
    board_hull,
    detect_checkerboard,
    median_corner_spacing,
)
from fishsense_data_processing_workflow_worker.lattice_overlay import (
    draw_lattice_overlay,
)
from fishsense_data_processing_workflow_worker.object_store import (
    open_object_store_client,
)

__all__ = ["render_checkerboard_lattice"]


def _unusable(payload, reason: str) -> tuple[CheckerboardLatticeRender, None]:
    return (
        CheckerboardLatticeRender(
            image_id=payload.image_id,
            checksum=payload.checksum,
            skip_reason=reason,
        ),
        None,
    )


def _render(
    raw_bytes: bytes,
    payload,
) -> tuple[CheckerboardLatticeRender, bytes | None]:
    """Sync helper run via `asyncio.to_thread` — decode, undistort, detect, draw.

    Returns `(render, jpeg_bytes)`, with `jpeg_bytes` None for a frame that
    produced no lattice. Never raises on an unusable frame: a dive holds dozens
    and one unreadable board is ordinary, so a raise would fail the whole dive
    over it.
    """
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(payload.camera_matrix, dtype=float),
        distortion_coefficients=np.array(payload.distortion_coefficients, dtype=float),
        camera_id=None,
    )
    rectified = RectifiedImage(RawImage(raw_bytes), intrinsics)

    detected = detect_checkerboard(
        rectified.data,
        max_rows=payload.target_rows,
        max_cols=payload.target_cols,
        square_size_m=payload.square_size_m,
    )
    if detected is None:
        return _unusable(payload, "no_usable_board")

    if not point_in_laser_region(
        float(payload.laser_x), float(payload.laser_y), board_hull(detected)
    ):
        return _unusable(payload, "dot_off_board")

    spacing = median_corner_spacing(detected.image_points, detected.rows, detected.cols)
    height, width = rectified.data.shape[:2]
    overlaid = draw_lattice_overlay(
        rectified.data,
        rows=detected.rows,
        cols=detected.cols,
        image_points=detected.image_points,
        caption=f"{detected.rows}x{detected.cols}  spacing {spacing:.1f}px",
    )

    return (
        CheckerboardLatticeRender(
            image_id=payload.image_id,
            checksum=payload.checksum,
            detected_rows=detected.rows,
            detected_cols=detected.cols,
            median_spacing_px=float(spacing),
            # Rounded to 1/100 px, which more than halves the payload: a full
            # 10x14 render goes from ~5.6 KB to ~2.6 KB, and a 256-frame dive's
            # workflow result from 1.44 MB to 0.66 MB — clear of Temporal's
            # 2 MB blob limit rather than sitting just under it.
            #
            # Safe because these corners never reach any geometry. The
            # calibration fit runs its own detection in
            # `detect_checkerboard_laser_point` and never reads this DTO; here
            # they only place drawing marks (already snapped to whole pixels by
            # `draw_lattice_overlay`) and Label Studio keypoints (percentages,
            # where 0.01 px of a 4000 px frame is 0.00025%).
            corners=[
                [round(float(x), 2), round(float(y), 2)]
                for x, y in detected.image_points
            ],
            width=int(width),
            height=int(height),
        ),
        encode_rectified_jpeg(overlaid),
    )


def _input_model():
    from fishsense_data_processing_workflow_worker.workflows import (
        verify_checkerboard_lattice_workflow as workflow,
    )

    return workflow.RenderCheckerboardLatticeInput


@activity.defn
async def render_checkerboard_lattice(payload) -> CheckerboardLatticeRender:
    """Draw this frame's detected lattice and upload it, or say why it has none."""
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    client = open_object_store_client()
    raw_bytes = await client.download_raw(payload.checksum)
    render, jpeg_bytes = await asyncio.to_thread(_render, raw_bytes, payload)

    if jpeg_bytes is None:
        activity.logger.info(
            "no lattice to render image_id=%d checksum=%s reason=%s",
            payload.image_id,
            payload.checksum,
            render.skip_reason,
        )
        return render

    await client.upload_processed_jpeg(
        CHECKERBOARD_LATTICE_JPEG_FOLDER, payload.checksum, jpeg_bytes
    )
    activity.logger.info(
        "rendered lattice image_id=%d grid=%dx%d spacing=%.1fpx",
        payload.image_id,
        render.detected_rows,
        render.detected_cols,
        render.median_spacing_px,
    )
    return render

"""Unit tests for the per-frame lattice render.

Fast counterparts to `test_lattice_verification_integration.py`: no Temporal,
no object store, no rawpy. What they cover that the integration tests do not is
the *admission* branches and the payload shape, both of which decide what a
labeler ends up judging.

`dot_off_board` is the branch that matters most and the integration suite
cannot reach it: the only raw fixture holds no board at all, so it lands on
`no_usable_board` before the dot is ever considered.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from fishsense_data_processing_workflow_worker.activities import (
    render_checkerboard_lattice as sut,
)
from fishsense_data_processing_workflow_worker.workflows.verify_checkerboard_lattice_workflow import (  # noqa: E501  pylint: disable=line-too-long
    RenderCheckerboardLatticeInput,
)

_K = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]

# The E4E board: 15 x 11 squares -> 14 x 10 interior corners.
_COLS_SQ, _ROWS_SQ = 15, 11
_COLS, _ROWS = _COLS_SQ - 1, _ROWS_SQ - 1
_SQUARE_PX = 60
_BORDER = 120


def _board() -> np.ndarray:
    height = _ROWS_SQ * _SQUARE_PX + 2 * _BORDER
    width = _COLS_SQ * _SQUARE_PX + 2 * _BORDER
    img = np.full((height, width), 255, np.uint8)
    for row in range(_ROWS_SQ):
        for col in range(_COLS_SQ):
            if (row + col) % 2 == 0:
                y_0, x_0 = _BORDER + row * _SQUARE_PX, _BORDER + col * _SQUARE_PX
                img[y_0 : y_0 + _SQUARE_PX, x_0 : x_0 + _SQUARE_PX] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _blank() -> np.ndarray:
    return np.full((800, 1000, 3), 200, np.uint8)


@pytest.fixture
def rectified_is(monkeypatch):
    """Substitute the decode so these stay unit tests.

    Only the decode: the detector, the overlay, the spacing and the DTO
    assembly below are all production code.
    """

    def _install(image: np.ndarray):
        class _FakeRectified:  # pylint: disable=too-few-public-methods
            def __init__(self, _raw, _intrinsics):
                self.data = image

        monkeypatch.setattr(sut, "RectifiedImage", _FakeRectified)
        monkeypatch.setattr(sut, "RawImage", lambda raw: raw)

    return _install


def _payload(image: np.ndarray, *, dot=None) -> RenderCheckerboardLatticeInput:
    height, width = image.shape[:2]
    dot_x, dot_y = dot if dot is not None else (width / 2, height / 2)
    return RenderCheckerboardLatticeInput(
        image_id=11,
        checksum="c" * 32,
        laser_x=dot_x,
        laser_y=dot_y,
        camera_matrix=_K,
        distortion_coefficients=[0.0] * 5,
        target_rows=_ROWS,
        target_cols=_COLS,
        square_size_m=0.042,
    )


def test_a_board_with_the_dot_on_it_renders(rectified_is):
    board = _board()
    rectified_is(board)

    render, jpeg = sut._render(b"raw", _payload(board))  # pylint: disable=protected-access

    assert render.skip_reason is None
    assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
    assert sorted((render.detected_rows, render.detected_cols)) == sorted(
        (_ROWS, _COLS)
    )


def test_a_dot_off_the_board_is_refused_and_uploads_nothing(rectified_is):
    """The branch the integration suite cannot reach.

    It keeps this study's population matched to the fit's: the calibration path
    refuses the same frames, so rendering one here would put a frame in front
    of a labeler that never contributed to the calibration under test.
    """
    board = _board()
    rectified_is(board)

    # Top-left corner, comfortably outside the detected grid's hull.
    render, jpeg = sut._render(  # pylint: disable=protected-access
        b"raw", _payload(board, dot=(2.0, 2.0))
    )

    assert render.skip_reason == "dot_off_board"
    assert jpeg is None
    assert render.corners is None


def test_a_frame_with_no_board_is_refused_and_uploads_nothing(rectified_is):
    blank = _blank()
    rectified_is(blank)

    render, jpeg = sut._render(b"raw", _payload(blank))  # pylint: disable=protected-access

    assert render.skip_reason == "no_usable_board"
    assert jpeg is None


def test_corners_are_rounded_to_hundredths_of_a_pixel(rectified_is):
    """A payload-size safeguard that nothing else would notice losing.

    Full float64 corners more than double the DTO — ~5.6 KB against ~2.6 KB for
    a 10x14 render — which puts a large dive's workflow result near Temporal's
    2 MB blob limit instead of comfortably under it. Safe to round because
    these corners never reach any geometry: the fit runs its own detection and
    never reads this DTO.
    """
    board = _board()
    rectified_is(board)

    render, _ = sut._render(b"raw", _payload(board))  # pylint: disable=protected-access

    assert render.corners
    for x, y in render.corners:
        assert x == round(x, 2)
        assert y == round(y, 2)


def test_the_render_reports_the_frame_it_actually_drew_on(rectified_is):
    """Dimensions must describe the rendered frame.

    Label Studio keypoints are percentages of the image, so a mismatch here
    scatters every mark — and a labeler would report that as a lattice fault,
    which is precisely the answer this study must not manufacture.
    """
    board = _board()
    rectified_is(board)

    render, jpeg = sut._render(b"raw", _payload(board))  # pylint: disable=protected-access
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)

    assert (render.width, render.height) == (decoded.shape[1], decoded.shape[0])
    assert (render.width, render.height) == (board.shape[1], board.shape[0])


def test_spacing_is_reported_and_matches_the_drawn_square(rectified_is):
    """The machine-readable counterpart of the labeler's verdict.

    A lattice at twice the true pitch reports twice the spacing, so the two
    answers can be checked against each other afterwards.
    """
    board = _board()
    rectified_is(board)

    render, _ = sut._render(b"raw", _payload(board))  # pylint: disable=protected-access

    assert render.median_spacing_px == pytest.approx(_SQUARE_PX, rel=0.02)

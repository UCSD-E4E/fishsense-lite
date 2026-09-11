"""End-to-end lattice-verification tests against the local devcontainer stack
(temporal + Garage object store).

Two tests, because one real fixture cannot cover both halves of this stage.

`stage2_sample.ORF` is the repo's only raw fixture and it is a dive frame with
no checkerboard in it. It exercises everything up to the detector — the rawpy
decode, the rectification, Garage, Temporal, the activity wiring — and then
lands on the *skip* path, which is genuinely worth pinning: a frame with no
board must upload nothing and say why, because a blank task in the labeling
project reads to a human as "the detector found nothing here" and would be
recorded as a verdict about the wrong thing.

The render path needs a board in frame. The second test keeps the whole stack
real — real Garage round-trip, real Temporal, real detector, real overlay, real
JPEG encode, real upload — and substitutes only the *decode*, handing the
activity a synthetic board instead of what rawpy would have produced. That one
substitution is exactly what the first test covers with real bytes, so between
them every step runs for real at least once.

Getting rid of the substitution means committing a ~13 MB checkerboard `.ORF`
pulled from the NAS. That is a deliberate call about carrying prod imagery in
the repo, not something to do by default, and the synthetic board is stricter
in one way that matters: its true grid is known, so the test can assert the
detector recovered *that* grid rather than merely some grid.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities import (
    render_checkerboard_lattice as activity_module,
)
from fishsense_data_processing_workflow_worker.activities.render_checkerboard_lattice import (  # noqa: E501  pylint: disable=line-too-long
    render_checkerboard_lattice,
)
from fishsense_data_processing_workflow_worker.workflows.verify_checkerboard_lattice_workflow import (  # noqa: E501  pylint: disable=line-too-long
    VerifyCheckerboardLatticeWorkflow,
)
from fishsense_shared import (
    CheckerboardCalibrationImage,
    VerifyCheckerboardLatticeInput,
)
from fishsense_shared.object_store import CHECKERBOARD_LATTICE_JPEG_FOLDER

# The sibling-helper import every stage's integration test uses. CI runs
# pylint from the repo root over a changed-file list, where `tests` is not
# an importable package -- resolvable only by pytest's rootdir, not pylint's.
# pylint: disable-next=import-error
from ._object_store_itest import BUCKET, make_s3_client, set_object_store_env

pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]

# The E4E board: 15 x 11 squares -> 14 x 10 interior corners, matching
# `test_checkerboard_detection.py`.
_BOARD_COLS_SQ, _BOARD_ROWS_SQ = 15, 11
_BOARD_COLS, _BOARD_ROWS = _BOARD_COLS_SQ - 1, _BOARD_ROWS_SQ - 1
_SQUARE_PX = 60
_SQUARE_SIZE_M = 0.042


def _temporal_target() -> str:
    host = os.environ.get("FISHSENSE_TEMPORAL_HOST", "temporal")
    port = os.environ.get("FISHSENSE_TEMPORAL_PORT", "7233")
    return f"{host}:{port}"


def _render_board() -> np.ndarray:
    """A flat checkerboard on a white ground, as BGR.

    The white border is not decoration — `findChessboardCornersSB` needs a
    quiet boundary around the pattern.
    """
    border = 120
    height = _BOARD_ROWS_SQ * _SQUARE_PX + 2 * border
    width = _BOARD_COLS_SQ * _SQUARE_PX + 2 * border
    img = np.full((height, width), 255, np.uint8)
    for row in range(_BOARD_ROWS_SQ):
        for col in range(_BOARD_COLS_SQ):
            if (row + col) % 2 == 0:
                y_0 = border + row * _SQUARE_PX
                x_0 = border + col * _SQUARE_PX
                img[y_0 : y_0 + _SQUARE_PX, x_0 : x_0 + _SQUARE_PX] = 0
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def raw_orf_bytes() -> bytes:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE.read_bytes()


@pytest.fixture
def configure_worker_settings(monkeypatch: pytest.MonkeyPatch):
    set_object_store_env(monkeypatch)
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


def _payload(checksum: str, **overrides) -> VerifyCheckerboardLatticeInput:
    kwargs = {
        "dive_id": -1,
        "camera_matrix": _K,
        "distortion_coefficients": _D,
        "target_rows": _BOARD_ROWS,
        "target_cols": _BOARD_COLS,
        "square_size_m": _SQUARE_SIZE_M,
        "images": [
            CheckerboardCalibrationImage(
                image_id=1,
                checksum=checksum,
                laser_x=0.0,
                laser_y=0.0,
            )
        ],
    }
    kwargs.update(overrides)
    return VerifyCheckerboardLatticeInput(**kwargs)


async def _run_workflow(payload: VerifyCheckerboardLatticeInput):
    client = await Client.connect(_temporal_target())
    task_queue = f"lattice-itest-{uuid.uuid4().hex}"
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[VerifyCheckerboardLatticeWorkflow],
        activities=[render_checkerboard_lattice],
    ):
        return await client.execute_workflow(
            VerifyCheckerboardLatticeWorkflow.run,
            payload,
            id=f"lattice-itest-{uuid.uuid4().hex}",
            task_queue=task_queue,
        )


def _key_absent(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except s3.exceptions.ClientError:
        return True
    return False


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_a_real_frame_with_no_board_uploads_nothing_and_says_why(
    raw_orf_bytes: bytes,
):
    """Real `.ORF` bytes through the whole stack, landing on the skip path.

    The uploaded-nothing half is the point. A JPEG written here would become a
    Label Studio task with no marks on it, and the only verdict a human could
    give is about the detector's hit rate rather than about the lattice.
    """
    checksum = f"itest-lattice-noboard-{uuid.uuid4().hex}"
    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{checksum}.ORF", Body=raw_orf_bytes)

    [render] = await _run_workflow(_payload(checksum))

    assert render.image_id == 1
    assert render.checksum == checksum
    assert render.skip_reason is not None
    assert render.corners is None
    assert _key_absent(s3, f"{CHECKERBOARD_LATTICE_JPEG_FOLDER}/{checksum}.JPG")


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_a_board_frame_renders_uploads_and_reports_its_grid(
    monkeypatch: pytest.MonkeyPatch,
):
    """The render path, with only the rawpy decode substituted.

    Everything else is live: the bytes make a real Garage round-trip, the
    workflow runs on a real Temporal server, and the detector, overlay, JPEG
    encode and upload are the production code paths.
    """
    board = _render_board()

    class _FakeRectified:  # pylint: disable=too-few-public-methods
        def __init__(self, _raw, _intrinsics):
            self.data = board

    # Patched on the activity module, which imports the name directly. The
    # activity still downloads the real object first, so the Garage read is
    # exercised even though its bytes are then ignored.
    monkeypatch.setattr(activity_module, "RectifiedImage", _FakeRectified)
    monkeypatch.setattr(activity_module, "RawImage", lambda raw: raw)

    checksum = f"itest-lattice-board-{uuid.uuid4().hex}"
    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{checksum}.ORF", Body=b"not-really-an-orf")

    # A dot at the board's centre, so `point_in_laser_region` admits the frame.
    height, width = board.shape[:2]
    payload = _payload(checksum, images=[
        CheckerboardCalibrationImage(
            image_id=1,
            checksum=checksum,
            laser_x=width / 2,
            laser_y=height / 2,
        )
    ])

    [render] = await _run_workflow(payload)

    assert render.skip_reason is None
    # The synthetic board's true grid is known, so this asserts the detector
    # recovered *that* grid rather than merely some grid. Orientation is free
    # (see `checkerboard_detection`), hence the sorted comparison.
    assert sorted((render.detected_rows, render.detected_cols)) == sorted(
        (_BOARD_ROWS, _BOARD_COLS)
    )
    assert len(render.corners) == render.detected_rows * render.detected_cols
    # Spacing is the diagnostic the study turns on, and here the answer is
    # known: adjacent interior corners are one square apart.
    assert render.median_spacing_px == pytest.approx(_SQUARE_PX, rel=0.02)

    out = s3.get_object(
        Bucket=BUCKET, Key=f"{CHECKERBOARD_LATTICE_JPEG_FOLDER}/{checksum}.JPG"
    )
    content = out["Body"].read()
    assert content[:2] == b"\xff\xd8"

    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    # The reported dimensions must match the image actually uploaded: Label
    # Studio keypoints are percentages of the frame, so a disagreement here
    # scatters every mark and a labeler would report the units bug as a
    # lattice fault — the one failure that would poison the study's answer.
    assert (render.width, render.height) == (decoded.shape[1], decoded.shape[0])
    # And the corners must lie inside it, for the same reason.
    xs = [x for x, _ in render.corners]
    ys = [y for _, y in render.corners]
    assert 0 <= min(xs) and max(xs) <= render.width
    assert 0 <= min(ys) and max(ys) <= render.height


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_the_render_never_lands_on_the_laser_stage_prefix(
    monkeypatch: pytest.MonkeyPatch,
):
    """Keys are by checksum, so a wrong prefix silently replaces the frame the
    laser labelers are looking at with a magenta lattice."""
    board = _render_board()

    class _FakeRectified:  # pylint: disable=too-few-public-methods
        def __init__(self, _raw, _intrinsics):
            self.data = board

    monkeypatch.setattr(activity_module, "RectifiedImage", _FakeRectified)
    monkeypatch.setattr(activity_module, "RawImage", lambda raw: raw)

    checksum = f"itest-lattice-prefix-{uuid.uuid4().hex}"
    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{checksum}.ORF", Body=b"not-really-an-orf")

    height, width = board.shape[:2]
    await _run_workflow(
        _payload(checksum, images=[
            CheckerboardCalibrationImage(
                image_id=1,
                checksum=checksum,
                laser_x=width / 2,
                laser_y=height / 2,
            )
        ])
    )

    assert _key_absent(s3, f"preprocess_jpeg/{checksum}.JPG")
    assert not _key_absent(
        s3, f"{CHECKERBOARD_LATTICE_JPEG_FOLDER}/{checksum}.JPG"
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_sample_limit_stops_frames_reaching_the_object_store(
    raw_orf_bytes: bytes,
):
    """The cap is enforced in the workflow, so a capped frame is never fetched.

    Worth an integration assertion rather than trusting the contract test: the
    cap exists to bound how much rendering and human labeling a study costs,
    and a cap that trimmed only the *results* would do neither.
    """
    kept = f"itest-lattice-kept-{uuid.uuid4().hex}"
    dropped = f"itest-lattice-dropped-{uuid.uuid4().hex}"
    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{kept}.ORF", Body=raw_orf_bytes)
    # Deliberately NOT uploaded: if the capped frame were fetched, the activity
    # would fail on a missing key and fail the workflow.

    payload = _payload(
        kept,
        sample_limit=1,
        images=[
            CheckerboardCalibrationImage(
                image_id=1, checksum=kept, laser_x=0.0, laser_y=0.0
            ),
            CheckerboardCalibrationImage(
                image_id=2, checksum=dropped, laser_x=0.0, laser_y=0.0
            ),
        ],
    )

    renders = await _run_workflow(payload)

    assert [r.image_id for r in renders] == [1]

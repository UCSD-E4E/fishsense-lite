"""Workflow contract + fit-activity tests for checkerboard calibration.

The detection kernel is covered by `test_checkerboard_detection.py` and the
geometry by `test_calibration_geometry.py`. What is left, and what these pin
down, is the wiring: that the fan-out hands every frame the board geometry it
was dispatched with, that unusable frames are dropped rather than failing the
dive, and that the fit refuses the same cases stage 13 refuses.
"""

from __future__ import annotations

import logging

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fishsense_shared import (
    CheckerboardCalibrationImage,
    CheckerboardObservation,
    PerformCheckerboardCalibrationInput,
)
from temporalio import activity
from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities import (
    fit_checkerboard_laser_extrinsics as fit_module,
)
from fishsense_data_processing_workflow_worker.workflows.perform_checkerboard_calibration_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DetectCheckerboardLaserPointInput,
    FitCheckerboardExtrinsicsInput,
    PerformCheckerboardCalibrationWorkflow,
)

CAMERA_MATRIX = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]
SQUARE_SIZE_M = 0.0254
TASK_QUEUE = "test-checkerboard-calibration"


def _input(*, images=2, **overrides):
    kwargs = {
        "dive_id": 488,
        "camera_id": 9,
        "camera_matrix": CAMERA_MATRIX,
        "distortion_coefficients": [0.0] * 5,
        "target_rows": 10,
        "target_cols": 14,
        "square_size_m": SQUARE_SIZE_M,
        "images": [
            CheckerboardCalibrationImage(
                image_id=100 + n,
                checksum=f"{n:032d}",
                laser_x=600.0 + n,
                laser_y=500.0,
            )
            for n in range(images)
        ],
    }
    kwargs.update(overrides)
    return PerformCheckerboardCalibrationInput(**kwargs)


# ---------- the fan-out ----------


@pytest.mark.asyncio
async def test_every_frame_is_dispatched_with_the_board_geometry():
    """The pitch travels in the payload, once per frame, from the dispatch.

    It is the only thing setting the scale of every length this calibration
    will later produce, so it must not be re-read anywhere downstream where a
    replay could pick up a different value than the run was started with.
    """
    seen: list[DetectCheckerboardLaserPointInput] = []

    @activity.defn(name="detect_checkerboard_laser_point")
    async def _detect(payload: DetectCheckerboardLaserPointInput):
        seen.append(payload)
        return CheckerboardObservation(
            image_id=payload.image_id,
            point=[0.01, 0.02, 1.4],
            laser_x=payload.laser_x,
            laser_y=payload.laser_y,
            detected_rows=10,
            detected_cols=14,
        )

    @activity.defn(name="fit_checkerboard_laser_extrinsics")
    async def _fit(_payload: FitCheckerboardExtrinsicsInput) -> int:
        return 77

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[PerformCheckerboardCalibrationWorkflow],
            activities=[_detect, _fit],
        ):
            result = await env.client.execute_workflow(
                PerformCheckerboardCalibrationWorkflow.run,
                _input(images=3),
                id="wf-checkerboard-geometry",
                task_queue=TASK_QUEUE,
            )

    assert result == 77
    # A set, not a list: the fan-out is `asyncio.gather`, so completion order
    # is not dispatch order. What must hold is that every frame is dispatched
    # exactly once, with its own dot and the dive's board geometry.
    assert sorted(p.image_id for p in seen) == [100, 101, 102]
    assert {(p.target_rows, p.target_cols) for p in seen} == {(10, 14)}
    assert {p.square_size_m for p in seen} == {SQUARE_SIZE_M}
    assert {p.image_id: p.laser_x for p in seen} == {
        100: 600.0,
        101: 601.0,
        102: 602.0,
    }


@pytest.mark.asyncio
async def test_unusable_frames_reach_the_fit_and_do_not_fail_the_dive():
    """A frame with no detectable board is ordinary, not an error.

    The fit sees every observation, including the empty ones — it is the
    single place that decides whether enough of them survived, so the count it
    reports is over the frames actually attempted.
    """
    @activity.defn(name="detect_checkerboard_laser_point")
    async def _detect(payload: DetectCheckerboardLaserPointInput):
        usable = payload.image_id != 101
        return CheckerboardObservation(
            image_id=payload.image_id,
            point=[0.01, 0.02, 1.4] if usable else None,
            laser_x=payload.laser_x,
            laser_y=payload.laser_y,
        )

    received: list[FitCheckerboardExtrinsicsInput] = []

    @activity.defn(name="fit_checkerboard_laser_extrinsics")
    async def _fit(payload: FitCheckerboardExtrinsicsInput) -> int:
        received.append(payload)
        return 5

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[PerformCheckerboardCalibrationWorkflow],
            activities=[_detect, _fit],
        ):
            await env.client.execute_workflow(
                PerformCheckerboardCalibrationWorkflow.run,
                _input(images=3),
                id="wf-checkerboard-partial",
                task_queue=TASK_QUEUE,
            )

    assert len(received) == 1
    observations = received[0].observations
    assert len(observations) == 3
    assert [o.point is None for o in observations] == [False, True, False]


# ---------- the fit ----------


def _observation(image_id: int, point, *, laser_x=600.0, laser_y=500.0):
    return CheckerboardObservation(
        image_id=image_id, point=point, laser_x=laser_x, laser_y=laser_y
    )


#: A realistic fitted origin: 10.4 cm from the camera centre, which is where
#: every sound calibration in the fleet sits. It used to be (0.01, 0.02) — a
#: 2.2 cm baseline no rig has — and that was harmless until
#: `check_baseline_plausible` started refusing physically implausible fits, at
#: which point the fixture was asking the activity to persist exactly what the
#: gate exists to stop.
_GOOD_ORIGIN_XY = (0.0624, 0.0832)


def _fake_calibrate_laser(_points):
    """Stand in for the Rust Atanasov kernel.

    Returns the 2-vector origin it really returns — z is implicit — so the
    padding the activity does stays under test rather than being assumed.
    """
    return np.array(_GOOD_ORIGIN_XY), np.array([0.0, 0.0, 1.0])


def _fit_input(observations):
    return FitCheckerboardExtrinsicsInput(
        dive_id=488,
        camera_id=9,
        camera_matrix=CAMERA_MATRIX,
        observations=observations,
    )


@pytest.mark.asyncio
async def test_fit_refuses_below_the_shared_threshold():
    """One threshold, imported from stage 13 rather than restated.

    A third copy that drifts from the cohort's `MIN_SLATE_LASER_POINTS` is the
    wedge this repo keeps rediscovering: cohort offers the dive, activity
    refuses it, nothing is written, and it is re-selected hourly forever.
    """
    payload = _fit_input(
        [_observation(100, [0.0, 0.0, 1.4]), _observation(101, None)]
    )

    with pytest.raises(ValueError, match="insufficient checkerboard laser points"):
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, payload
        )


@pytest.mark.asyncio
async def test_fit_persists_the_extrinsics_it_computed(monkeypatch):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.put_laser_extrinsics = AsyncMock(return_value=31)
    monkeypatch.setattr(fit_module, "get_fs_client", lambda: fs)
    monkeypatch.setattr(fit_module, "_calibrate_laser", _fake_calibrate_laser)
    monkeypatch.setattr(fit_module, "check_fit_self_consistency", lambda *a, **k: None)

    payload = _fit_input(
        [
            _observation(100, [0.0, 0.0, 1.40]),
            _observation(101, [0.0, 0.0, 1.60], laser_x=620.0),
            _observation(102, None),
        ]
    )

    assert (
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, payload
        )
        == 31
    )
    dive_id, extrinsics = fs.dives.put_laser_extrinsics.await_args.args
    assert dive_id == 488
    assert extrinsics.camera_id == 9
    # The Rust kernel returns a 2-vector origin with z implicit; stage 13 pads
    # it the same way, and the SDK surface is a 3-vector.
    assert list(extrinsics.laser_position) == [*_GOOD_ORIGIN_XY, 0.0]


@pytest.mark.asyncio
async def test_fit_does_not_persist_when_the_gate_rejects(monkeypatch):
    """The self-consistency gate is the last thing between a bad fit and prod.

    A mixed dot population shipped a calibration whose length errors reached
    +137% downstream on prod dive 77. The gate raises; nothing is written.
    """
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.put_laser_extrinsics = AsyncMock()
    monkeypatch.setattr(fit_module, "get_fs_client", lambda: fs)
    monkeypatch.setattr(fit_module, "_calibrate_laser", _fake_calibrate_laser)

    def _reject(*_args, **_kwargs):
        raise ValueError("fit disagrees with its own dots")

    monkeypatch.setattr(fit_module, "check_fit_self_consistency", _reject)

    payload = _fit_input(
        [_observation(100, [0.0, 0.0, 1.4]), _observation(101, [0.0, 0.0, 1.6])]
    )

    with pytest.raises(ValueError, match="disagrees"):
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, payload
        )
    fs.dives.put_laser_extrinsics.assert_not_awaited()


def test_the_threshold_is_the_stage_13_one():
    """Imported, not restated — asserted so a later edit cannot fork it."""
    from fishsense_data_processing_workflow_worker.activities import (
        perform_laser_calibration_activity as stage13,
    )

    assert fit_module.MIN_LASER_POINTS is stage13.MIN_LASER_POINTS


# ---------- the skip tally ----------
#
# A dive fitted from half its frames and one fitted from all of them look
# identical afterwards, and the first is telling you something. Same reason
# the laser-depth stage counts `skipped_invalid_geometry`.


@pytest.mark.asyncio
async def test_the_fit_tallies_why_frames_were_dropped(monkeypatch, caplog):
    """The reasons reach the log, not just the count."""
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.put_laser_extrinsics = AsyncMock(return_value=7)
    monkeypatch.setattr(fit_module, "get_fs_client", lambda: fs)
    monkeypatch.setattr(fit_module, "_calibrate_laser", _fake_calibrate_laser)
    monkeypatch.setattr(fit_module, "check_fit_self_consistency", lambda *a, **k: None)

    def _skipped(image_id: int, reason: str) -> CheckerboardObservation:
        return CheckerboardObservation(
            image_id=image_id,
            point=None,
            laser_x=1.0,
            laser_y=1.0,
            skip_reason=reason,
        )

    payload = _fit_input(
        [
            _observation(100, [0.0, 0.0, 1.40]),
            _observation(101, [0.0, 0.0, 1.60], laser_x=620.0),
            _skipped(102, "dot_off_board"),
            _skipped(103, "dot_off_board"),
            _skipped(104, "no_usable_board"),
        ]
    )

    with caplog.at_level(logging.INFO):
        result = await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, payload
        )

    assert result == 7
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "usable=2 of 5" in logged
    assert "'dot_off_board': 2" in logged
    assert "'no_usable_board': 1" in logged


@pytest.mark.asyncio
async def test_the_refusal_says_why_the_frames_went():
    """A dive that falls short must not just say "not enough points".

    The remedy differs entirely by reason: every frame `dot_off_board` means
    the laser was not on the board and the capture is the problem, while
    `no_usable_board` means the board was not found and the target link or the
    frames are. Without the tally an operator sees the same message for both.
    """
    payload = _fit_input(
        [
            _observation(100, [0.0, 0.0, 1.4]),
            CheckerboardObservation(
                image_id=101, point=None, laser_x=1.0, laser_y=1.0,
                skip_reason="dot_off_board",
            ),
        ]
    )

    with pytest.raises(ValueError, match="dot_off_board"):
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, payload
        )

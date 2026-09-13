"""Both fit activities must actually call the gates, before persisting.

`test_calibration_baseline_gate.py`, `test_calibration_describes_dive.py` and
`test_calibration_observation_geometry.py` test the gates as pure functions,
which says nothing about whether anything calls them. Deleting a call site —
or moving it below `put_laser_extrinsics` — leaves those suites entirely green
while the bad calibration lands in the database and becomes borrowable by
sibling dives through `calibration_dive_id`.

So these drive the activities end to end with a fit that the gate must refuse,
and assert two things: nothing was written, and the refusal reached Temporal as
**non-retryable**. The second matters as much as the first. Both gates are
deterministic functions of the observations the run was dispatched with, so a
retry re-derives the same answer; left retryable, Temporal reschedules until
the child's 2 h execution timeout, holding the parent, keeping the dive's raw
scratch alive and skipping two hourly firings.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from fishsense_data_processing_workflow_worker.activities import (
    fit_checkerboard_laser_extrinsics as fit_module,
)
from fishsense_data_processing_workflow_worker.workflows.perform_checkerboard_calibration_workflow import (  # noqa: E501  pylint: disable=line-too-long
    FitCheckerboardExtrinsicsInput,
)
from fishsense_shared import CheckerboardObservation

from ._calibration_fixtures import make_fit_client

CAMERA_MATRIX = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]

#: 2.35 cm — dive 522's real fitted baseline, and the worst of the eight.
_IMPLAUSIBLE_XY = (0.0141, 0.0188)
#: 10.4 cm, where every sound calibration in the fleet sits.
_PLAUSIBLE_XY = (0.0624, 0.0832)


def _observations(n: int = 12) -> list[CheckerboardObservation]:
    """Enough well-spread observations that nothing else refuses first."""
    return [
        CheckerboardObservation(
            image_id=100 + i,
            point=[0.01 * i, 0.02, 1.0 + 0.1 * i],
            laser_x=600.0 + 40 * i,
            laser_y=500.0 + 30 * i,
        )
        for i in range(n)
    ]


def _payload() -> FitCheckerboardExtrinsicsInput:
    return FitCheckerboardExtrinsicsInput(
        dive_id=522,
        camera_id=9,
        camera_matrix=CAMERA_MATRIX,
        observations=_observations(),
    )


def _patch_fit(monkeypatch, origin_xy, *, dive_offset_px: float = 0.0) -> MagicMock:
    """Force the kernel's answer and capture whether anything was persisted.

    The dive's own dots are built on the projection of the ray the fake kernel
    returns, so `check_calibration_describes_dive` passes unless a test asks
    for a `dive_offset_px` — the honest stand-in for "the whole dive agrees".
    """

    def _fake_calibrate(_points):
        # Axis pointing straight down the optical axis keeps the
        # self-consistency check satisfied, so only the baseline gate can fire.
        return np.array(origin_xy), np.array([0.0, 0.0, 1.0])

    monkeypatch.setattr(fit_module, "_calibrate_laser", _fake_calibrate)
    monkeypatch.setattr(
        fit_module, "check_fit_self_consistency", lambda *a, **k: None
    )

    client = make_fit_client(
        (*origin_xy, 0.0),
        (0.0, 0.0, 1.0),
        CAMERA_MATRIX,
        dive_offset_px=dive_offset_px,
    )
    monkeypatch.setattr(fit_module, "get_fs_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_checkerboard_fit_refuses_an_implausible_baseline(monkeypatch):
    client = _patch_fit(monkeypatch, _IMPLAUSIBLE_XY)

    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, _payload()
        )

    assert excinfo.value.non_retryable
    assert "baseline" in str(excinfo.value).lower()
    client.dives.put_laser_extrinsics.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkerboard_fit_persists_a_plausible_baseline(monkeypatch):
    """The complement, so the gate cannot be 'passed' by refusing everything."""
    client = _patch_fit(monkeypatch, _PLAUSIBLE_XY)

    result = await ActivityEnvironment().run(
        fit_module.fit_checkerboard_laser_extrinsics, _payload()
    )

    assert result == 7
    client.dives.put_laser_extrinsics.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_refusal_names_the_dive(monkeypatch):
    """A wedged dive is found from the log line, so it has to say which."""
    _patch_fit(monkeypatch, _IMPLAUSIBLE_XY)

    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, _payload()
        )

    assert "522" in str(excinfo.value)


@pytest.mark.asyncio
async def test_checkerboard_fit_refuses_a_fit_the_dive_disagrees_with(monkeypatch):
    """The gate the other three cannot stand in for.

    Baseline and self-consistency both look only at the fit's own
    observations, so a board burst that is internally perfect but was shot
    with the laser in a different state from the dive's fish frames passes
    them both. Prod dive 347's calibration sits 9.4 px off its own 321 dots
    and was caught by nothing.
    """
    client = _patch_fit(monkeypatch, _PLAUSIBLE_XY, dive_offset_px=40.0)

    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, _payload()
        )

    assert excinfo.value.non_retryable
    assert "522" in str(excinfo.value)
    client.dives.put_laser_extrinsics.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkerboard_fit_refuses_a_single_distance_burst(monkeypatch):
    """A board burst at one distance determines the ray's direction no better
    than one frame does, however many corners it detects — and it is the
    geometry under which every other gate abstains."""
    client = _patch_fit(monkeypatch, _PLAUSIBLE_XY)
    payload = _payload()
    payload.observations = [
        CheckerboardObservation(
            image_id=100 + i,
            point=[0.01 * i, 0.02, 1.40],
            laser_x=600.0 + 40 * i,
            laser_y=500.0 + 30 * i,
        )
        for i in range(12)
    ]

    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, payload
        )

    assert excinfo.value.non_retryable
    assert "522" in str(excinfo.value)
    client.dives.put_laser_extrinsics.assert_not_awaited()


def test_the_slate_fit_gates_before_it_persists():
    """Source-order check on stage 13, which needs live SDK data to drive.

    Crude, and deliberately so: the property is that the gate call precedes
    `put_laser_extrinsics` in the function body. Asserting it by execution
    would mean standing up the whole slate-label and PnP pass, and the failure
    this guards against — someone moving the call below the write — is exactly
    a textual reordering.
    """
    import inspect

    from fishsense_data_processing_workflow_worker.activities import (
        perform_laser_calibration_activity as slate_module,
    )

    source = inspect.getsource(slate_module)
    write_at = source.index("put_laser_extrinsics")
    for gate in (
        "check_observation_geometry(",
        "check_fit_self_consistency(",
        "check_baseline_plausible(laser_position)",
        "check_calibration_describes_dive(",
    ):
        assert source.index(gate) < write_at, gate

"""Both fit activities must actually record a refusal, not just raise.

`_record_refusal` swallows every exception on purpose — losing the real
`CalibrationImplausibleError` to an HTTP error would be strictly worse than
the pre-existing behaviour. The cost of that choice is that a broken call is
invisible: a renamed SDK method would make the whole feature a silent no-op
with nothing but an ERROR log, and the existing workflow tests would still
pass because they never patch the client at all.

So these assert the call happens, with the dive and a reason, for each of the
deterministic refusals — and that it does NOT happen on the happy path, since
recording a refusal for a dive that calibrated fine would park a healthy dive.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

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

CAMERA_MATRIX = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]
#: 2.35 cm — dive 522's real fitted baseline, which the gate must refuse.
IMPLAUSIBLE_XY = (0.0141, 0.0188)
#: 10.4 cm, where every sound calibration in the fleet sits.
PLAUSIBLE_XY = (0.0624, 0.0832)


def _observations(n: int):
    return [
        CheckerboardObservation(
            image_id=100 + i,
            point=[0.01 * i, 0.02, 1.0 + 0.1 * i],
            laser_x=600.0 + 40 * i,
            laser_y=500.0 + 30 * i,
        )
        for i in range(n)
    ]


def _payload(n: int = 12):
    return FitCheckerboardExtrinsicsInput(
        dive_id=522,
        camera_id=9,
        camera_matrix=CAMERA_MATRIX,
        observations=_observations(n),
    )


def _patch(monkeypatch, origin_xy):
    """Force the kernel's answer; capture the refusal call."""
    monkeypatch.setattr(
        fit_module,
        "_calibrate_laser",
        lambda _p: (np.array(origin_xy), np.array([0.0, 0.0, 1.0])),
    )
    monkeypatch.setattr(fit_module, "check_fit_self_consistency", lambda *a, **k: None)

    client = MagicMock()
    client.dives.set_calibration_refused = AsyncMock(return_value=522)
    client.dives.put_laser_extrinsics = AsyncMock(return_value=7)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(fit_module, "get_fs_client", lambda: ctx)
    return client


@pytest.mark.asyncio
async def test_an_implausible_baseline_is_recorded(monkeypatch):
    client = _patch(monkeypatch, IMPLAUSIBLE_XY)

    with pytest.raises(ApplicationError):
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, _payload()
        )

    client.dives.set_calibration_refused.assert_awaited_once()
    dive_id, reason = client.dives.set_calibration_refused.await_args.args
    assert dive_id == 522
    assert "baseline" in reason.lower()


@pytest.mark.asyncio
async def test_too_few_observations_is_recorded(monkeypatch):
    client = _patch(monkeypatch, PLAUSIBLE_XY)

    with pytest.raises(ApplicationError):
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, _payload(n=1)
        )

    client.dives.set_calibration_refused.assert_awaited_once()
    dive_id, reason = client.dives.set_calibration_refused.await_args.args
    assert dive_id == 522
    assert "insufficient" in reason.lower()


@pytest.mark.asyncio
async def test_a_successful_fit_records_no_refusal(monkeypatch):
    """The complement — otherwise the feature could 'pass' by parking everything."""
    client = _patch(monkeypatch, PLAUSIBLE_XY)

    result = await ActivityEnvironment().run(
        fit_module.fit_checkerboard_laser_extrinsics, _payload()
    )

    assert result == 7
    client.dives.set_calibration_refused.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failure_to_record_does_not_mask_the_refusal(monkeypatch):
    """The reason `_record_refusal` swallows, stated as a test.

    The api and the data-worker deploy from separate auto-deploy PRs, so the
    worker can run ahead of an api without the route. Losing the real
    `CalibrationImplausibleError` — and `non_retryable` with it — would make
    Temporal re-run the whole fit until its timeout.
    """
    client = _patch(monkeypatch, IMPLAUSIBLE_XY)
    client.dives.set_calibration_refused = AsyncMock(
        side_effect=RuntimeError("404 Not Found")
    )

    with pytest.raises(ApplicationError) as excinfo:
        await ActivityEnvironment().run(
            fit_module.fit_checkerboard_laser_extrinsics, _payload()
        )

    assert excinfo.value.non_retryable
    assert "baseline" in str(excinfo.value).lower()

"""A Label Studio 429 must never be mistaken for a missing project.

`sync_label_studio_project` probes `ls.projects.get(project_id)` and treated
any `ApiError` as "project is gone" — log a warning, return. A 429 throttle
raises the same `ApiError` as a 404, so a rate-limited probe silently skipped
the project AND returned before the cursor write, repeating every hour with
no error surfaced.

Observed in prod: project 274633 sat at 86/86 tasks labeled in Label Studio,
0 completed in the DB, and no cursor row had ever been written. Each hourly
cycle probes ~46 projects across four label kinds; LS starts throttling
partway through and everything after that point is dropped — which is why it
looked non-deterministic (58 and 60 synced, 59 never did).
"""

from __future__ import annotations

# Tests exercise the internal throttle helpers directly.
# pylint: disable=protected-access

from unittest.mock import MagicMock

import pytest
from label_studio_sdk.core import ApiError
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import utils as sut


def _throttle(seconds: int = 48) -> ApiError:
    return ApiError(
        status_code=429,
        body={"detail": f"Request was throttled. Expected available in {seconds} seconds."},
    )


def test_throttle_wait_honours_the_hint():
    assert sut._throttle_wait_seconds(_throttle(48)) == 50.0  # hint + margin


def test_throttle_wait_falls_back_when_hint_is_unparseable():
    err = ApiError(status_code=429, body={"detail": "Request was throttled."})
    assert sut._throttle_wait_seconds(err) == sut._THROTTLE_DEFAULT_WAIT_SECONDS


@pytest.mark.parametrize("code", [404, 403, 500])
def test_non_throttle_errors_are_not_treated_as_throttles(code):
    assert sut._throttle_wait_seconds(ApiError(status_code=code, body={})) is None


async def test_a_404_still_means_missing():
    """Genuinely-gone projects must still be skipped — legacy ids 57-117
    all 404 on the hosted instance and would otherwise fail every run."""
    ls = MagicMock()
    ls.projects.get.side_effect = ApiError(status_code=404, body={})

    ran = await ActivityEnvironment().run(sut._ls_project_exists, ls, 73, "headtail")
    assert ran is False


async def test_a_throttle_retries_then_succeeds(monkeypatch):
    """The 274633 case: throttled first, fine on retry — must NOT be skipped."""
    sleeps: list[float] = []

    async def _no_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(sut.asyncio, "sleep", _no_sleep)
    ls = MagicMock()
    ls.projects.get.side_effect = [_throttle(48), MagicMock()]

    ran = await ActivityEnvironment().run(sut._ls_project_exists, ls, 274633, "headtail")
    assert ran is True
    assert sleeps == [50.0], "should have honoured the retry-after hint"


async def test_persistent_throttle_raises_rather_than_skipping(monkeypatch):
    """Failing loudly is the point: a raise leaves the cursor unadvanced so
    the next run retries, instead of silently dropping the project."""
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(sut.asyncio, "sleep", _no_sleep)
    ls = MagicMock()
    ls.projects.get.side_effect = _throttle(60)

    with pytest.raises(RuntimeError, match="still throttling"):
        await ActivityEnvironment().run(sut._ls_project_exists, ls, 274633, "headtail")
    assert ls.projects.get.call_count == sut._THROTTLE_MAX_ATTEMPTS

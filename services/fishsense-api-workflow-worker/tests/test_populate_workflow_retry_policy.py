"""The populate activity's retries must be bounded, and the command order kept.

With Temporal's default policy an activity retries without limit inside its
`schedule_to_close_timeout`. Dive 424's populate reached attempt 10 over 1107
seconds and left 23 copies of three frames. The import helper closes the path
that duplicated, but a bound is what keeps any residual window from compounding.

The order assertion is the replay contract: `_populate.create_then_populate`
replaced four inlined copies, and a workflow in flight at deploy only replays
cleanly if it still emits create-then-populate, in that order, with the same
timeouts.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

import pytest
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _populate as sut

STAGES = ["laser", "species", "headtail", "dive_slate"]


@pytest.fixture(name="calls")
def _calls(monkeypatch):
    seen: List[Dict[str, Any]] = []

    async def fake_execute_activity(name, **kwargs):
        seen.append({"name": name, **kwargs})
        return 7 if name.startswith("create_") else 42

    monkeypatch.setattr(workflow, "execute_activity", fake_execute_activity)
    return seen


@pytest.mark.parametrize("stage", STAGES)
async def test_emits_create_then_populate_in_order(stage, calls):
    result = await sut.create_then_populate(stage, 393)

    assert [c["name"] for c in calls] == [
        f"create_{stage}_label_studio_project_activity",
        f"populate_{stage}_label_studio_project_activity",
    ]
    assert calls[0]["args"] == (393,)
    assert calls[1]["args"] == (393, 7), "populate takes the created project id"
    assert result == 42


@pytest.mark.parametrize("stage", STAGES)
async def test_timeouts_are_unchanged(stage, calls):
    await sut.create_then_populate(stage, 1)

    assert calls[0]["schedule_to_close_timeout"] == timedelta(minutes=5)
    assert calls[1]["schedule_to_close_timeout"] == timedelta(minutes=30)
    assert calls[1]["heartbeat_timeout"] == timedelta(minutes=2)


@pytest.mark.parametrize("stage", STAGES)
async def test_populate_retries_are_bounded(stage, calls):
    await sut.create_then_populate(stage, 1)

    policy = calls[1].get("retry_policy")
    assert policy is not None, "unlimited retries are what compounded duplicates"
    assert policy.maximum_attempts == sut.POPULATE_MAX_ATTEMPTS
    assert 1 < policy.maximum_attempts <= 5


@pytest.mark.parametrize("stage", STAGES)
async def test_the_retry_window_still_absorbs_a_label_studio_blip(stage, calls):
    """Capping attempts alone would have been a regression.

    Temporal's default backoff starts at 1s and doubles, so three attempts
    cover about three seconds. A half-minute LS blip would then fail the
    populate child and, through `dispatch_populate`, the preprocess parent —
    which in the slate parent runs *before* `clear_slate_reprocess_flags`, so
    the dive would keep its reprocess flag and re-stage its raw frames from
    NAS on the next firing.
    """
    await sut.create_then_populate(stage, 1)
    policy = calls[1]["retry_policy"]

    assert policy.initial_interval >= timedelta(seconds=30)
    assert policy.maximum_interval is not None

    window = timedelta()
    interval = policy.initial_interval
    for _ in range(policy.maximum_attempts - 1):
        window += interval
        interval = min(
            interval * policy.backoff_coefficient, policy.maximum_interval
        )
    assert window >= timedelta(minutes=5), "must ride out an ordinary LS blip"
    assert window <= timedelta(minutes=15), "but stay far short of the old 30"

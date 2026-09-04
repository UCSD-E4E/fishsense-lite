"""The gate child workflow declares the shared timeout budget, not its own.

The three values are a cross-worker contract (`fishsense_shared`), because the
api-worker parents' `execution_timeout` has to outlast what this workflow asks
for. Spelling literals here instead would let the two drift silently -- which
is how the 15-minute `schedule_to_close_timeout` copied from the laser-label
validation workflow survived into v2.19.0 and stalled the backlog drain.

Driven by monkeypatching `workflow.execute_activity` rather than by standing up
a Temporal environment: the assertion is about the *arguments* of one dispatch,
and a stub keeps that legible.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from fishsense_shared import (
    GATE_ACTIVITY_TIMEOUT,
    GATE_EXECUTION_TIMEOUT,
    GATE_QUEUE_WAIT_TIMEOUT,
    LaserAutoAcceptSummary,
)
from temporalio import workflow

from fishsense_data_processing_workflow_worker.workflows import (
    evaluate_laser_auto_accept_workflow as sut,
)


@pytest.fixture(name="dispatch")
def _dispatch(monkeypatch):
    """Run the workflow body with a recording stub in place of Temporal."""
    calls: dict = {}

    async def _record(activity_name, **kwargs):
        calls["activity_name"] = activity_name
        calls.update(kwargs)
        return LaserAutoAcceptSummary(
            dive_id=7,
            eligible=True,
            reason=None,
            n_points=30,
            auto_accepted=7,
            verdicts={"auto_accepted": 7},
        )

    monkeypatch.setattr(workflow, "execute_activity", _record)
    asyncio.run(sut.EvaluateLaserAutoAcceptWorkflow().run(7))
    return calls


def test_it_dispatches_the_gate_activity(dispatch):
    assert dispatch["activity_name"] == "evaluate_laser_auto_accept_activity"
    assert dispatch["args"] == (7,)


def test_queue_wait_is_bounded_separately_from_execution(dispatch):
    """The bug being fixed: with only `schedule_to_close` set, Temporal spends
    the whole budget on the queue and reports `ScheduleToStart timeout`. An
    explicit `schedule_to_start_timeout` is what makes the two independently
    readable -- and independently tunable when the CPU queue gets busier."""
    assert dispatch["schedule_to_start_timeout"] == GATE_QUEUE_WAIT_TIMEOUT
    assert dispatch["start_to_close_timeout"] == GATE_EXECUTION_TIMEOUT


def test_the_close_bound_leaves_room_for_a_late_start(dispatch):
    """`schedule_to_close` covers queue wait *plus* execution, so an attempt
    that gets its slot at the last moment still gets its full run."""
    assert dispatch["schedule_to_close_timeout"] == GATE_ACTIVITY_TIMEOUT


def test_the_fetch_still_heartbeats(dispatch):
    """Unchanged, and load-bearing next to a tight `start_to_close`: it turns a
    silent hang in `get_laser_predictions` into a diagnosable timeout rather
    than a ten-minute wait."""
    assert dispatch["heartbeat_timeout"] == timedelta(minutes=1)

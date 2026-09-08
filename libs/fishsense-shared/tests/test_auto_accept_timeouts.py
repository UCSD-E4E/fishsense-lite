"""The auto-accept gate's timeout budget.

These numbers are a cross-worker contract, which is why they live in
`fishsense_shared` rather than in either worker: the data-worker's child
workflow declares how long its activity may wait and run, and *two* api-worker
parents declare how long they will wait for that child. The three have to be
ordered correctly or the wrong timeout fires first and the logs name the wrong
thing.

They were re-derived from a prod failure. The gate shipped with a single
15-minute `schedule_to_close_timeout` copied from the laser-label validation
workflow, which conflates queue wait with execution. On 2026-09-04 two of the
drain's first three firings died with `ScheduleToStart timeout` after sitting
behind multi-GB rawpy decodes on the four-slot CPU queue -- the fit itself
takes under a second. Only one dive drained in three hours.
"""

from datetime import timedelta

from fishsense_shared import auto_accept_timeouts as sut

# What the api-worker parents declare for their *other* steps, from
# `workflows/_dispatch.py`: `select_dive` and `wake_data_worker` are 5 min
# each, `run_sdk_activity` (the apply step) is 15 min.
_PARENT_OTHER_STEPS = timedelta(minutes=5) + timedelta(minutes=5) + timedelta(minutes=15)

# `worker.py` registers `evaluate-laser-auto-accept-workflow-schedule` with
# this `run_timeout`, and `ensure_schedule` refuses to update an existing
# schedule in place -- so a change here would need an operator to delete the
# schedule and let the next worker start recreate it. Fitting inside the value
# already deployed is what keeps this a code-only change.
_PARENT_RUN_TIMEOUT = timedelta(hours=1)


def test_the_activity_budget_is_queue_wait_plus_execution():
    """`schedule_to_close` must be the *sum*, not a single conflated number.

    Set to anything less and an attempt that finally gets a slot near the end
    of its patience is cut off mid-fit -- the queue wait would silently eat
    into the execution budget, which is the bug this module exists to stop
    being expressible.
    """
    assert (
        sut.GATE_ACTIVITY_TIMEOUT
        == sut.GATE_QUEUE_WAIT_TIMEOUT + sut.GATE_EXECUTION_TIMEOUT
    )


def test_the_queue_wait_outlasts_the_value_that_failed_in_prod():
    """15 minutes is the number that died twice on 2026-09-04. The point of
    this change is patience for the queue, so anything at or under that is a
    regression to the failing configuration."""
    assert sut.GATE_QUEUE_WAIT_TIMEOUT > timedelta(minutes=15)


def test_the_execution_bound_stays_tight():
    """Only the *queue wait* was wrong. The fit is sub-second and the fetch is
    heartbeated at one minute, so a run that has started and is still going at
    ten minutes is stuck, not slow -- keeping this tight is what makes the
    heartbeat diagnostic useful."""
    assert sut.GATE_EXECUTION_TIMEOUT == timedelta(minutes=10)


def test_the_child_outlives_the_activity_it_runs():
    """The child exists only to run this one activity. If the child's
    `execution_timeout` fired first, the failure would surface as an opaque
    `ChildWorkflowError` with no timeout type -- exactly the reading that made
    the prod failure take a log dig to diagnose. The activity's own timeout
    must always be the one that fires, so it names which bound was hit."""
    assert sut.GATE_CHILD_EXECUTION_TIMEOUT > sut.GATE_ACTIVITY_TIMEOUT


def test_the_whole_gate_fits_inside_the_parents_deployed_run_timeout():
    """Worst case for a firing is every declared step burning its full budget.
    That has to fit in the `run_timeout` already registered in prod, because
    `ensure_schedule` will not update a live schedule -- so a budget that
    needed a bigger `run_timeout` would ship as a silent no-op until an
    operator deleted the schedule by hand."""
    worst_case = sut.GATE_CHILD_EXECUTION_TIMEOUT + _PARENT_OTHER_STEPS

    assert worst_case <= _PARENT_RUN_TIMEOUT

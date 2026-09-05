"""Timeout budget for the laser auto-accept gate.

These live here for the same reason `task_queues.py` does: they are an
agreement *between* the two workers. The data-worker's
`EvaluateLaserAutoAcceptWorkflow` declares how long its activity may wait for a
slot and how long it may then run; **two** api-worker parents --
`EvaluateLaserAutoAcceptParentWorkflow` (the hourly backlog drain) and
`PredictLaserImagesParentWorkflow` (the inline path) -- declare how long they
will wait for that child. Three declarations, one budget, and nothing in either
worker's source makes the third visible from the other two.

**Separating queue wait from execution is the whole point.** The gate shipped
in v2.19.0 with a single `schedule_to_close_timeout=15m`, copied from the
laser-label validation workflow, which conflates the two. The fit is a RANSAC
line through one dive's dots and takes under a second; what actually consumes
the budget is waiting for one of the four activity slots on
`fishsense_data_processing_queue`, which the preprocess stages hold through
multi-GB rawpy decodes. On 2026-09-04 that killed two of the drain's first
three firings with `ScheduleToStart timeout` -- one dive drained in three
hours, against a backlog of 65.

So the queue wait is the number that had to grow, and the execution bound is
the number that had to stay put. A run that has *started* and is still going at
ten minutes is stuck, not slow, and the one-minute heartbeat on the fetch is
what says so.

**The ceiling is not a matter of taste.** `worker.py` registers the drain's
schedule with `run_timeout=1h`, and `ensure_schedule` deliberately refuses to
update an existing schedule in place -- so raising it would ship as a silent
no-op until an operator deleted the schedule and let the next worker start
recreate it. Every value here is sized to fit inside the `run_timeout` that is
already deployed, alongside the parent's other declared steps (5 min selector +
5 min data-worker wake + 15 min apply = 25 min). That is what makes this a
code-only change. `test_auto_accept_timeouts.py` pins the arithmetic.

Patience is cheap here and a lost firing is not: a firing that times out does
not fall back to a smaller dive, it drops the whole hour, and the backlog
drains at one dive per hour at best.
"""

from datetime import timedelta

# How long the gate activity may sit on a busy CPU queue before giving up.
# Raised from an effective 15 min after the 2026-09-04 failures above.
GATE_QUEUE_WAIT_TIMEOUT = timedelta(minutes=20)

# How long it may run once it has a slot. Unchanged -- the fit is sub-second
# and the dominant cost is `get_laser_predictions` over Traefik on a large
# dive, which heartbeats.
GATE_EXECUTION_TIMEOUT = timedelta(minutes=10)

# `schedule_to_close`: the sum, never less. Anything smaller lets a long queue
# wait eat into the execution budget and cut off a fit that had just started.
GATE_ACTIVITY_TIMEOUT = GATE_QUEUE_WAIT_TIMEOUT + GATE_EXECUTION_TIMEOUT

# What both api-worker parents pass as the child's `execution_timeout`. Must
# exceed `GATE_ACTIVITY_TIMEOUT` so the *activity's* timeout is the one that
# fires: it names which bound was hit, where a child-level timeout surfaces
# only as an opaque `ChildWorkflowError`.
GATE_CHILD_EXECUTION_TIMEOUT = timedelta(minutes=35)

__all__ = [
    "GATE_ACTIVITY_TIMEOUT",
    "GATE_CHILD_EXECUTION_TIMEOUT",
    "GATE_EXECUTION_TIMEOUT",
    "GATE_QUEUE_WAIT_TIMEOUT",
]

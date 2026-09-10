"""Shared body of the four `Populate<Stage>LabelStudioProjectWorkflow`s.

Laser, species, head/tail and dive-slate populate ran the same two steps in
the same order, in four files that differed only in two activity-name strings.
`duplicate-code` cannot see that — the names differ, so it is textual
divergence over identical logic, the shape most duplication in this repo takes
(see the `put_*_label` handlers). Keeping the play in one place is also what
let the bounded retry policy below land once rather than four times.

**This helper emits the same Temporal commands, in the same order, that the
inlined code did** — a workflow's command sequence is its replay contract, so
runs in flight at deploy still replay. Same reasoning, and same rule, as
`_dispatch.py`.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

__all__ = ["POPULATE_MAX_ATTEMPTS", "create_then_populate"]

# Bounded on purpose, where the default is unlimited-within-schedule_to_close:
# unlimited is what let dive 424 reach activity attempt 10 over 1107s and leave
# 23 copies of three frames.
#
# **The intervals matter as much as the cap.** Temporal's default backoff
# starts at 1s and doubles, so capping attempts alone would have collapsed the
# retry window from 30 minutes to about 3 seconds — a half-minute Label Studio
# blip would then fail the populate child and, through `dispatch_populate`,
# take the preprocess parent with it. Starting at 30s and allowing 5 attempts
# keeps roughly 8 minutes of cover, which rides out an ordinary blip while
# staying far short of the old window.
#
# Re-importing across those attempts is prevented in the activity, not here:
# the first import heartbeats a marker that Temporal hands to the next attempt
# (`populate_utils._IMPORT_ISSUED`), so a retry reconciles instead. Without
# that, widening this window would widen the duplication window with it.
POPULATE_MAX_ATTEMPTS = 5
_POPULATE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=30),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=POPULATE_MAX_ATTEMPTS,
)


async def create_then_populate(stage: str, dive_id: int) -> int:
    """Materialise `stage`'s per-dive project, then push its tasks.

    `stage` is the activity-name infix: `laser`, `species`, `headtail` or
    `dive_slate`. Returns the number of label rows written.
    """
    project_id = await workflow.execute_activity(
        f"create_{stage}_label_studio_project_activity",
        args=(dive_id,),
        schedule_to_close_timeout=timedelta(minutes=5),
    )
    return await workflow.execute_activity(
        f"populate_{stage}_label_studio_project_activity",
        args=(dive_id, project_id),
        schedule_to_close_timeout=timedelta(minutes=30),
        heartbeat_timeout=timedelta(minutes=2),
        retry_policy=_POPULATE_RETRY,
    )

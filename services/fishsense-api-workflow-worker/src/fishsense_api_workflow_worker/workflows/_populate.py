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

# Bounded on purpose, where the default is unlimited-within-schedule_to_close.
#
# The import helper no longer treats an unmaterialised import as an error, so
# the path that duplicated tasks is closed at source. One narrow window is left:
# an import lands, its tasks stay invisible past the poll budget, and then
# something *else* in the same attempt fails — the retry's dedup listing still
# cannot see those tasks and re-imports them. Unlimited attempts turned that
# window into 23 copies of a dive-424 frame (activity attempt 10, over 1107s).
# Three caps the damage at one extra copy while still absorbing the ordinary
# transient failure.
POPULATE_MAX_ATTEMPTS = 3


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
        retry_policy=RetryPolicy(maximum_attempts=POPULATE_MAX_ATTEMPTS),
    )

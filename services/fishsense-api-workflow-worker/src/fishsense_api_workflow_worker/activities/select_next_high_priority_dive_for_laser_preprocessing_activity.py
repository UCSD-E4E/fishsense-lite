"""Activity to pick the next HIGH-priority dive that needs stage 0.1
laser preprocessing.

Cohort: HIGH priority + at least one *canonical* image carrying no
non-sentinel `LaserLabel` row (i.e. none with a real
`label_studio_project_id`). The SQL predicate lives in the api's
`select-next/laser-preprocessing` endpoint — see
`select_next_for_laser_preprocessing` in `dive_cohort_controller`.

The cohort used to be "HIGH priority + no `LaserExtrinsics` row yet",
and this module's docstring and log line still said so long after the
code changed. That predicate tied stage 0.1 to a *downstream* gate it
doesn't advance: a dive whose laser side was finished but whose slate
side blocked stage-13 calibration kept getting re-selected every hour
with no work for the resolver to return. The log line was the worse
half — it told an operator the selector had asked about
`laser_extrinsics` when it had asked about label rows.

Lives on the api-worker so the SDK call runs on the interior docker
network (no authentik / cross-host hop). The selector is a single SDK
call that returns the cohort answer in one query — the prior shape was
a `dives.get()` plus a sequential `get_laser_extrinsics(dive_id)` per
HIGH-priority dive, which timed out the activity's schedule_to_close on
backlogs of a few hundred.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_laser_preprocessing_activity() -> int | None:
    return await select_next_dive(
        "laser preprocessing",
        lambda fs: fs.dives.select_next_for_laser_preprocessing(),
    )

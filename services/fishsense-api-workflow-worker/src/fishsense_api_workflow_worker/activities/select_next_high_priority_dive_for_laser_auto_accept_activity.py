"""Activity to pick the next HIGH-priority dive whose laser predictions the
auto-accept gate has never judged.

Cohort: HIGH-priority + at least one canonical image whose `LaserPrediction`
carries a dot and has a NULL `gate_verdict` (see
`select_next_for_laser_auto_accept` in the api's dive_cohort_controller).

Exists because the gate normally runs off the back of the predict parent, and
only when the predict child returned *new* predictions — so a dive that was
already fully predicted never produced results and never got judged. That was
3,711 rows across ~65 dives when the gate shipped.

Single SDK call — the cohort answer is one query on the api side.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_laser_auto_accept_activity() -> int | None:
    return await select_next_dive(
        "laser auto-accept judging",
        lambda fs: fs.dives.select_next_for_laser_auto_accept(),
    )

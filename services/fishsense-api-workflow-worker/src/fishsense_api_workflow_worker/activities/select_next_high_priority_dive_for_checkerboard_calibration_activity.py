"""Activity to pick the next HIGH-priority dive needing checkerboard calibration.

Cohort: HIGH priority + `calibration_target_id` set + no `LaserExtrinsics` of
its own + at least `MIN_SLATE_LASER_POINTS` (=2) canonical images carrying a
live laser dot. The sibling of stage 13's selector, for the dives whose
calibration frames show a printed board rather than one of the 11 `DiveSlate`
templates.

Note what an observation is here. Stage 13 counts hand-clicked `DiveSlateLabel`
rows whose image also carries a dot, because a human supplied its
correspondences; a checkerboard's corners are found by the data-worker at run
time, so the only thing SQL can count is the dot.

That makes this cohort a wider over-approximation than stage 13's — it cannot
know whether a board will actually be found — and the consequence is the
head-of-line shape prod dive 347 produced: refused by the activity, re-selected
hourly, blocking every higher-id dive. Both remedies are operator-side and
already exist (clear the dive's calibration target, or park it at
`Priority.NONE`); the endpoint's docstring names them.

The selector is a single SDK call; the SQL predicate lives in the api's
`select-next/checkerboard-laser-calibration` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_checkerboard_calibration_activity() -> (
    int | None
):
    return await select_next_dive(
        "checkerboard calibration",
        lambda fs: fs.dives.select_next_for_checkerboard_laser_calibration(),
    )

"""Activity to pick the next HIGH-priority dive that needs stage 13
laser calibration.

Cohort: HIGH priority + has `dive_slate_id` set + no `LaserExtrinsics`
row yet + at least `MIN_COMPLETED_SLATE_LABELS` (=2) completed
`DiveSlateLabel` rows. The minimum-2 floor matches the data-worker
activity's `MIN_LASER_POINTS = 2` precondition; selecting a dive with
fewer than two completed slate labels would dispatch a child that
raises `ValueError` and re-fires every hour, since
`put_laser_extrinsics` never gets written.

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/laser-calibration` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_laser_calibration_activity() -> int | None:
    return await select_next_dive(
        "laser calibration",
        lambda fs: fs.dives.select_next_for_laser_calibration(),
    )

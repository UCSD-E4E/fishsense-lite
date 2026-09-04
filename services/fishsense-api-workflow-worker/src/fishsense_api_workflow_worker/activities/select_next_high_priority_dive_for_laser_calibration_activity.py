"""Activity to pick the next HIGH-priority dive that needs stage 13
laser calibration.

Cohort: HIGH priority + has `dive_slate_id` set + no `LaserExtrinsics`
row yet + at least `MIN_SLATE_LASER_POINTS` (=2) usable slate-laser
*observations* — completed `DiveSlateLabel` rows whose image also carries
a live (non-superseded, x/y set) `LaserLabel`.

**Observations, not slate labels.** This docstring used to say the floor
counted completed `DiveSlateLabel` rows and that it "matches the
data-worker activity's `MIN_LASER_POINTS = 2` precondition". It did not:
the activity keeps a slate label only when `get_laser_label(image_id)`
returns a dot, so a dive can clear a slate-label count and still have no
observations. The comment asserted a parity the code never had — the
same way the taxonomy predicates drifted behind their cross-references.

The consequence is the one the old text described accurately and then
failed to prevent: dispatching a child that raises `ValueError` and
re-fires every hour, since `put_laser_extrinsics` never gets written.
Prod dive 347 — 18 completed slate labels, one live dot, the rest
superseded by the breach recovery — did exactly that, hourly, and
blocked dives 427 and 436 behind it.

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

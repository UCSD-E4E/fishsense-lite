"""Activity to pick the next HIGH-priority dive that needs stage 14
fish measurement.

Cohort: HIGH priority + a resolvable `LaserExtrinsics` (its own, or a
sibling's via `Dive.calibration_dive_id`) + at least one *measurable* image
not yet measured under that calibration. "Measurable" mirrors what
`measure_fish_activity` actually attempts: a top-three species label whose
image carries a valid laser label, a valid head/tail label, and — for real
fish, not rigid models — a LABEL_STUDIO cluster.

"Not yet measured under that calibration" is stricter than "has no
Measurement", and deliberately: a length is a function of the extrinsics
behind its depth, so replacing a dive's calibration invalidates it. A row
naming a superseded calibration (or, for anything written before
`Measurement.laser_extrinsics_id` existed, naming none) puts the dive back in
the cohort, and the activity recomputes it in place rather than skipping.

The parent workflow is scheduled hourly at +40 (2026-07-17). This docstring
previously described it as deliberately unscheduled because
`measure_fish_activity` was non-idempotent and the cohort keyed on
`cluster.fish_id IS NULL` — a predicate that could never go false. Both were
fixed on that date; the description was not.

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/measure-fish` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_measure_fish_activity() -> int | None:
    return await select_next_dive(
        "measurement",
        lambda fs: fs.dives.select_next_for_measure_fish(),
    )

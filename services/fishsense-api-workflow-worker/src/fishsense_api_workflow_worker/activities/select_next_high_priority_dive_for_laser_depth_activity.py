"""Activity to pick the next HIGH-priority dive needing laser depths.

Cohort: HIGH priority + a resolvable `LaserExtrinsics` (its own, or a
sibling's via `Dive.calibration_dive_id`) + at least one canonical image whose
validated laser label has no *current* `LaserDepth` — no row at all, or one
computed from a label or a calibration that has since been replaced.

Broader than stage 14's cohort by design: the distance to the laser dot is
knowable for any frame whose laser was validated, with no need for head/tail
labels, a cluster, or a measurable species. That is the whole reason the depth
is stored per image rather than hung off `Measurement`.

Because the cohort is keyed on the *inputs* rather than on mere existence, it
doubles as the backfill: a dive whose depths predate a recalibration re-enters
on its own and drops out once recomputed. No one-shot script, and nothing to
remember to run.

The selector is a single SDK call; the SQL predicate lives in the api's
`select-next/laser-depth` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_laser_depth_activity() -> int | None:
    return await select_next_dive(
        "laser depth",
        lambda fs: fs.dives.select_next_for_laser_depth(),
    )

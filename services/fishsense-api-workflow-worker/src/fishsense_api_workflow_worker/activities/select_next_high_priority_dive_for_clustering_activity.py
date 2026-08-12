"""Activity to pick the next HIGH-priority dive needing stage-1
dive-frame clustering.

Cohort: HIGH priority + has at least one image carrying a *valid*
LaserLabel (completed=True, superseded=False, both x/y populated)
AND has zero PREDICTION DiveFrameCluster rows. The clustering
output is the prerequisite for stage-2 species preprocessing, so
firing as soon as labelers + the validator sign off on lasers
keeps stage 2 ready to go on the next hourly tick.

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/dive-frame-clustering` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_clustering_activity() -> int | None:
    return await select_next_dive(
        "dive-frame clustering",
        lambda fs: fs.dives.select_next_for_dive_frame_clustering(),
    )

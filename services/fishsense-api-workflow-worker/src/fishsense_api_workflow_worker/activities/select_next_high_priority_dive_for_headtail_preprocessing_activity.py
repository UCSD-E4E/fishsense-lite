"""Activity to pick the next HIGH-priority dive that needs stage 5.1
head/tail preprocessing.

Cohort: HIGH priority + has at least one image carrying a *valid*
LaserLabel (completed=True, superseded=False, both x/y populated)
whose head/tail label is missing (matches
`populate_headtail_label_studio_project_activity`'s predicate).

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/headtail-preprocessing` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.cohort_selection import (
    select_next_dive,
)


@activity.defn
async def select_next_high_priority_dive_for_headtail_preprocessing_activity() -> int | None:
    return await select_next_dive(
        "headtail preprocessing",
        lambda fs: fs.dives.select_next_for_headtail_preprocessing(),
    )

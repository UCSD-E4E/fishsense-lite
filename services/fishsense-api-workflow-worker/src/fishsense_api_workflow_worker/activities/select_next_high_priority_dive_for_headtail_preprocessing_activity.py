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

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_headtail_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_headtail_preprocessing()

    if dive_id is None:
        activity.logger.info(
            "no HIGH-priority dives needing headtail preprocessing"
        )
    else:
        activity.logger.info(
            "next HIGH-priority dive needing headtail preprocessing: dive_id=%d",
            dive_id,
        )
    return dive_id

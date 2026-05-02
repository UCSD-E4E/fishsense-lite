"""Activity to pick the next HIGH-priority dive that needs stage 9
slate preprocessing.

Cohort: HIGH priority + has a `dive_slate_id` set + has at least one
species label with `content_of_image == 'Slate, Laser on slate'` whose
dive_slate_label is missing or not yet completed (matches
`populate_dive_slate_label_studio_project_activity`'s predicate).

The selector is a single SDK call; the SQL predicate lives in the
api's `select-next/slate-preprocessing` endpoint.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_slate_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_slate_preprocessing()

    if dive_id is None:
        activity.logger.info(
            "no HIGH-priority dives needing slate preprocessing"
        )
    else:
        activity.logger.info(
            "next HIGH-priority dive needing slate preprocessing: dive_id=%d",
            dive_id,
        )
    return dive_id

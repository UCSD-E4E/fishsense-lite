"""Activity to pick the next HIGH-priority dive needing slate predictions.

Cohort: HIGH-priority + `dive_slate_id` set + at least one slate frame
(`SpeciesLabel.content_of_image='Slate, Laser on slate'`) with no
`SlatePrediction` and no *completed* `DiveSlateLabel` (see
`select_next_for_slate_prediction` in the api's dive_controller). The CPU
slate-detector stage seeds a prediction per such frame; the dive-slate populate
step then serves it as a Label Studio pre-annotation (assisted review).

Single SDK call — the cohort answer is one query on the api side.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_slate_prediction_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_slate_prediction()

    if dive_id is None:
        activity.logger.info(
            "no HIGH-priority dives needing slate predictions; nothing to predict"
        )
    else:
        activity.logger.info(
            "next HIGH-priority dive needing slate predictions: dive_id=%d",
            dive_id,
        )
    return dive_id

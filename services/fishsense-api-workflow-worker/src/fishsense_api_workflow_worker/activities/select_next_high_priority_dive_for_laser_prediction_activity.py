"""Activity to pick the next HIGH-priority dive needing laser predictions.

Cohort: HIGH-priority + at least one image with no `LaserPrediction` and no
non-sentinel `LaserLabel` (see `select_next_for_laser_prediction` in the api's
dive_controller). The GPU laser-detector stage seeds a prediction per such
image; the laser populate step then serves it as a Label Studio pre-annotation
(assisted review).

Single SDK call — the cohort answer is one query on the api side.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_laser_prediction_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_laser_prediction()

    if dive_id is None:
        activity.logger.info(
            "no HIGH-priority dives needing laser predictions; nothing to predict"
        )
    else:
        activity.logger.info(
            "next HIGH-priority dive needing laser predictions: dive_id=%d",
            dive_id,
        )
    return dive_id

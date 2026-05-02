"""Activity to pick the next HIGH-priority dive that needs stage 0.1
laser preprocessing.

Cohort definition: HIGH priority + no `LaserExtrinsics` row yet (same
target the stage-13 calibration script uses, see
`scripts/dry_run_stage13.py` in the data-worker package). Producing
JPEGs for these dives is the work that unblocks downstream
calibration.

Lives on the api-worker so the SDK call runs on the orchestrator's
docker network (no authentik / cross-host hop). The selector itself
is a single SDK call that returns the cohort answer in one query —
the prior shape was a `dives.get()` plus a sequential
`get_laser_extrinsics(dive_id)` per HIGH-priority dive, which timed
out the activity's schedule_to_close on backlogs of a few hundred.
"""

from __future__ import annotations

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_laser_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dive_id = await fs.dives.select_next_for_laser_preprocessing()

    if dive_id is None:
        activity.logger.info(
            "no HIGH-priority dives without laser_extrinsics; nothing to preprocess"
        )
    else:
        activity.logger.info(
            "next HIGH-priority dive needing laser preprocessing: dive_id=%d",
            dive_id,
        )
    return dive_id

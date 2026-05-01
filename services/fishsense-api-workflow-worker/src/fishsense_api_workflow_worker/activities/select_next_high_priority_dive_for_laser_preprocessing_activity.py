"""Activity to pick the next HIGH-priority dive that needs stage 0.1
laser preprocessing.

Cohort definition: HIGH priority + no `LaserExtrinsics` row yet (same
target the stage-13 calibration script uses, see
`scripts/dry_run_stage13.py` in the data-worker package). Producing
JPEGs for these dives is the work that unblocks downstream
calibration.

Lives on the api-worker so the SDK call runs on the orchestrator's
docker network (no authentik / cross-host hop). Returns the lowest
dive_id in the cohort, or None when the cohort is empty.
"""

from __future__ import annotations

from fishsense_api_sdk.models.priority import Priority
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def select_next_high_priority_dive_for_laser_preprocessing_activity() -> (
    int | None
):
    async with get_fs_client() as fs:
        dives = await fs.dives.get() or []
        candidates = [
            d
            for d in dives
            if d.priority == Priority.HIGH and d.id is not None
        ]
        candidates.sort(key=lambda d: d.id)

        for dive in candidates:
            extrinsics = await fs.dives.get_laser_extrinsics(dive.id)
            if extrinsics is None:
                activity.logger.info(
                    "next HIGH-priority dive needing laser preprocessing: dive_id=%d",
                    dive.id,
                )
                return dive.id

        activity.logger.info(
            "no HIGH-priority dives without laser_extrinsics; nothing to preprocess"
        )
        return None

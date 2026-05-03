"""Activity returning dive IDs whose laser labeling is fully complete."""

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def get_dives_with_complete_laser_labeling_activity() -> List[int]:
    """Get dive IDs where every non-superseded laser label is completed.

    Used by `SyncLabelStudioLaserLabelsWorkflow` after the per-project
    sync to gate the per-dive RANSAC-line validation step on a stable
    label population.
    """
    async with get_fs_client() as fs:
        dive_ids = await fs.labels.get_dives_with_complete_laser_labeling()

    activity.logger.info(
        "Found %d dives with complete laser labeling", len(dive_ids)
    )
    return dive_ids

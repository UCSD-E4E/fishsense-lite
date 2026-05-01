"""Activity to get unique Label Studio project IDs with dive-slate labels."""

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def get_dive_slate_label_studio_project_ids_activity() -> List[int]:
    """Activity to get unique Label Studio project IDs with dive-slate labels."""
    async with get_fs_client() as fs:
        label_studio_project_ids = (
            await fs.labels.get_dive_slate_label_studio_project_ids()
        )

    activity.logger.info(
        "Found %d unique Label Studio project IDs with dive-slate labels",
        len(label_studio_project_ids),
    )

    return label_studio_project_ids

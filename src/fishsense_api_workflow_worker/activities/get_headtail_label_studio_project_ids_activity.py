"""Activity to get unique Label Studio project IDs with headtail labels."""

from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.label_utils import get_headtail_labels
from fishsense_api_workflow_worker.activities.utils import get_fs_client


@activity.defn
async def get_headtail_label_studio_project_ids_activity() -> List[int]:
    """Activity to get unique Label Studio project IDs with headtail labels."""
    async with get_fs_client() as fs:
        headtail_labels = await get_headtail_labels(fs)

    label_studio_project_ids = list(
        {
            label.label_studio_project_id
            for label in headtail_labels
            if label.label_studio_project_id is not None
        }
    )
    activity.logger.info(
        f"Found {len(label_studio_project_ids)} unique Label Studio project IDs with "
        + "headtail labels"
    )

    return label_studio_project_ids

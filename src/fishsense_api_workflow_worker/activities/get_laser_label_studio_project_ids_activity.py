"""Activity to get unique Label Studio project IDs with laser labels."""

import asyncio
from typing import List

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_client


@activity.defn
async def get_laser_label_studio_project_ids_activity() -> List[int]:
    """Activity to get unique Label Studio project IDs with laser labels."""
    async with get_client() as fs:
        dives = await fs.dives.get_canonical()

        laser_labels = await asyncio.gather(
            *(fs.labels.get_laser_labels(dive.id) for dive in dives)
        )
        laser_labels = [label for sublist in laser_labels for label in sublist]

    label_studio_project_ids = list(
        {
            label.label_studio_project_id
            for label in laser_labels
            if label.label_studio_project_id is not None
        }
    )
    activity.logger.info(
        f"Found {len(label_studio_project_ids)} unique Label Studio project IDs with laser labels"
    )

    return label_studio_project_ids

"""Activity to sync laser labels for a Label Studio project."""

import json
from typing import Any

from fishsense_api_sdk.client import Client
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import (
    SYNC_CONCURRENCY,
    sync_label_studio_project,
)

LASER_LABEL_KEY_NAMES = ["kp-1", "laser"]

__all__ = [
    "sync_laser_labels_for_label_studio_project_activity",
    "SYNC_CONCURRENCY",
]


async def __update_laser_label(fs: Client, task: Any) -> None:
    laser_label = await fs.labels.get_laser_label(label_studio_id=task.id)

    # Skip if no laser label exists for this task
    if laser_label is None:
        return

    if task.annotators:
        user = await fs.users.get_by_label_studio_id(task.annotators[-1])
        laser_label.user_id = user.id

    laser_label.label_studio_json = json.loads(task.json())
    laser_label.updated_at = task.updated_at
    laser_label.completed = task.is_labeled

    if len(task.annotations) > 0:
        for laser_label_key in LASER_LABEL_KEY_NAMES:
            laser_label_section = [
                r
                for r in task.annotations[0]["result"]
                if r["from_name"] == laser_label_key
            ]
            laser_label_section = (
                laser_label_section[0] if len(laser_label_section) > 0 else None
            )

            if laser_label_section is not None:
                break

        if laser_label_section is not None:
            original_width = laser_label_section["original_width"]
            original_height = laser_label_section["original_height"]

            laser_x = laser_label_section["value"]["x"] * original_width / 100
            laser_y = laser_label_section["value"]["y"] * original_height / 100
            laser_label_str = laser_label_section["value"]["keypointlabels"][0]

            laser_label.x = laser_x
            laser_label.y = laser_y
            laser_label.label = laser_label_str

    await fs.labels.put_laser_label(laser_label.image_id, laser_label)


@activity.defn
async def sync_laser_labels_for_label_studio_project_activity(project_id: int):
    """Activity to sync laser labels for a Label Studio project."""
    await sync_label_studio_project(project_id, __update_laser_label, kind="laser")

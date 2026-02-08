"""Activity to sync laser labels for a Label Studio project."""

import asyncio
import json
from typing import Any

from fishsense_api_sdk.client import Client
from label_studio_sdk.core import ApiError
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client, get_ls_client
from fishsense_api_workflow_worker.exception_group_error_logging import (
    ExceptionGroupErrorLogging,
)

LASER_LABEL_KEY_NAMES = ["kp-1", "laser"]


async def __update_laser_label(fs: Client, task: Any):
    laser_label = await fs.labels.get_laser_label(label_studio_id=task.id)

    # Skip if no laser label exists for this task
    if laser_label is None:
        return

    if task.annotators:
        user = await fs.users.get_by_label_studio_id(task.annotators[-1])
        laser_label.user_id = user.id

    laser_label.label_studio_json = json.dumps(task.json())
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
    # pylint: disable=duplicate-code

    ls = get_ls_client()

    # Handle case where project does not exist
    try:
        _ = await asyncio.to_thread(ls.projects.get, project_id)
    except ApiError as e:
        activity.logger.warning(f"Error fetching project {project_id}: {e}")
        return

    tasks = await asyncio.to_thread(ls.tasks.list, project=project_id)

    async with get_fs_client() as fs:
        async with asyncio.TaskGroup() as tg:
            with ExceptionGroupErrorLogging(activity.logger):
                for task in tasks:
                    if activity.is_cancelled():
                        activity.logger.info(
                            "Activity cancelled, stopping sync for project %d",
                            project_id,
                        )
                        return

                    tg.create_task(__update_laser_label(fs, task))

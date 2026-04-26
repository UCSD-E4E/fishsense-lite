"""Activity to sync headtail labels for a Label Studio project."""

import asyncio
import json
from typing import Any

from fishsense_api_sdk.client import Client
from label_studio_sdk.core import ApiError
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client, get_ls_client
from fishsense_shared import ExceptionGroupErrorLogging


async def __update_headtail_label(fs: Client, task: Any):
    headtail_label = await fs.labels.get_headtail_label(label_studio_id=task.id)

    # Skip if no headtail label exists for this task
    if headtail_label is None:
        return

    if task.annotators:
        user = await fs.users.get_by_label_studio_id(task.annotators[-1])
        headtail_label.user_id = user.id

    headtail_label.label_studio_json = json.loads(task.json())
    headtail_label.completed = task.is_labeled
    headtail_label.updated_at = task.updated_at

    if len(task.annotations) > 0:
        headtail_label_sections = [
            r for r in task.annotations[0]["result"] if r["from_name"] == "kp-1"
        ]
        head_label_section = next(
            (
                section
                for section in headtail_label_sections
                if section["value"]["keypointlabels"][0] == "Snout"
            ),
            None,
        )
        tail_label_section = next(
            (
                section
                for section in headtail_label_sections
                if section["value"]["keypointlabels"][0] == "Fork"
            ),
            None,
        )

        if head_label_section is not None and tail_label_section is not None:
            original_width = head_label_section["original_width"]
            original_height = head_label_section["original_height"]

            head_x = head_label_section["value"]["x"] * original_width / 100
            head_y = head_label_section["value"]["y"] * original_height / 100
            tail_x = tail_label_section["value"]["x"] * original_width / 100
            tail_y = tail_label_section["value"]["y"] * original_height / 100

            headtail_label.head_x = head_x
            headtail_label.head_y = head_y
            headtail_label.tail_x = tail_x
            headtail_label.tail_y = tail_y

        await fs.labels.put_headtail_label(headtail_label.image_id, headtail_label)


@activity.defn
async def sync_headtail_labels_for_label_studio_project_activity(project_id: int):
    """Sync headtail labels for a Label Studio project."""
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
        async with ExceptionGroupErrorLogging(activity.logger):
            async with asyncio.TaskGroup() as tg:
                for task in tasks:
                    if activity.is_cancelled():
                        activity.logger.info(
                            "Activity cancelled, stopping sync for project %d",
                            project_id,
                        )
                        return

                    tg.create_task(__update_headtail_label(fs, task))

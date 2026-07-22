"""Activity to sync headtail labels for a Label Studio project."""

import json
from typing import Any

from fishsense_api_sdk.client import Client
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import (
    SYNC_CONCURRENCY,
    resolve_annotator_user,
    sync_label_studio_project,
)

__all__ = [
    "sync_headtail_labels_for_label_studio_project_activity",
    "SYNC_CONCURRENCY",
]


async def __update_headtail_label(fs: Client, task: Any) -> None:
    headtail_label = await fs.labels.get_headtail_label(label_studio_id=task.id)

    # Skip if no headtail label exists for this task
    if headtail_label is None:
        return

    # Attribution is best-effort: hosted LS returns `annotators` as
    # dicts rather than ints, and mis-handling that used to 422 and
    # kill the whole project's sync. See resolve_annotator_user.
    user = await resolve_annotator_user(fs, task)
    if user is not None:
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
    """Activity to sync headtail labels for a Label Studio project."""
    await sync_label_studio_project(project_id, __update_headtail_label, kind="headtail")

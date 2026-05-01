"""Utility functions for activities."""

import asyncio
from typing import Any, Awaitable, Callable

from fishsense_api_sdk.client import Client
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.core import ApiError
from temporalio import activity

from fishsense_api_workflow_worker.config import settings
from fishsense_shared import ExceptionGroupErrorLogging

SYNC_CONCURRENCY = 8


def get_ls_client():
    """Get Label Studio client.
    Returns:
        LabelStudio: Label Studio client
    """
    return LabelStudio(
        base_url=settings.label_studio.url, api_key=settings.label_studio.api_key
    )


def get_fs_client() -> Client:
    """Get Fishsense API client.

    Returns:
        Client: Fishsense API client
    """
    return Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    )


async def sync_label_studio_project(
    project_id: int,
    update_fn: Callable[[Client, Any], Awaitable[None]],
    *,
    concurrency: int = SYNC_CONCURRENCY,
) -> None:
    """Run a per-task `update_fn` across every Label Studio task in
    `project_id`, with bounded concurrency and per-task heartbeat.

    The fan-out is capped at `concurrency` to keep both the SDK
    connection pool and the API itself from being saturated by a
    project with thousands of tasks. Each task heartbeats on completion
    so Temporal sees liveness within the activity's heartbeat_timeout.

    A missing LS project is logged and treated as a no-op (warning, not
    error) so a single rogue project_id doesn't fail the whole sync.
    """
    ls = get_ls_client()

    try:
        _ = await asyncio.to_thread(ls.projects.get, project_id)
    except ApiError as e:
        activity.logger.warning(f"Error fetching project {project_id}: {e}")
        return

    tasks = await asyncio.to_thread(ls.tasks.list, project=project_id)
    sem = asyncio.Semaphore(concurrency)

    async with get_fs_client() as fs:

        async def _run(task: Any) -> None:
            async with sem:
                await update_fn(fs, task)
                activity.heartbeat()

        async with ExceptionGroupErrorLogging(activity.logger):
            async with asyncio.TaskGroup() as tg:
                for task in tasks:
                    if activity.is_cancelled():
                        activity.logger.info(
                            "Activity cancelled, stopping sync for project %d",
                            project_id,
                        )
                        return

                    tg.create_task(_run(task))

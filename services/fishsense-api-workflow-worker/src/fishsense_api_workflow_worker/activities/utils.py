"""Utility functions for activities."""

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.label_studio_sync_cursor import LabelStudioSyncCursor
from label_studio_sdk.client import LabelStudio
from label_studio_sdk.core import ApiError
from temporalio import activity

from fishsense_api_workflow_worker.config import settings
from fishsense_shared import ExceptionGroupErrorLogging

SYNC_CONCURRENCY = 8

# Heartbeat cadence for the long initial LS-task listing on a backlog
# project. Comfortably under the 2m heartbeat_timeout in the workflow
# (see sync_label_studio_*_workflow.py).
_LISTING_HEARTBEAT_INTERVAL_SECONDS = 30.0


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


def _coerce_updated_at(value: Any) -> datetime | None:
    """Pull a `datetime` out of an LS task's `updated_at`.

    The LS SDK sometimes hands back ISO strings, sometimes datetimes.
    Anything else (None, garbage) yields None — the caller treats that
    as "no comparable timestamp," which conservatively means the task
    is processed regardless of the cursor.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


async def _list_ls_tasks_with_heartbeat(
    ls: LabelStudio, project_id: int
) -> list[Any]:
    """Fully materialize an LS project's task list off the asyncio event
    loop while pumping heartbeats.

    `ls.tasks.list(...)` returns a lazy `SyncPager`; iterating it issues
    one synchronous HTTP call per page. On a backlog project this can
    page for the full activity timeout, so doing it on the main thread
    starves heartbeat flush and trips the 2m heartbeat_timeout. Page in
    a worker thread; heartbeat from the main thread on a fixed cadence.
    """

    async def _pump() -> None:
        try:
            while True:
                await asyncio.sleep(_LISTING_HEARTBEAT_INTERVAL_SECONDS)
                activity.heartbeat()
        except asyncio.CancelledError:
            return

    pump = asyncio.create_task(_pump())
    try:
        return await asyncio.to_thread(
            lambda: list(ls.tasks.list(project=project_id))
        )
    finally:
        pump.cancel()
        try:
            await pump
        except asyncio.CancelledError:
            pass


async def sync_label_studio_project(
    project_id: int,
    update_fn: Callable[[Client, Any], Awaitable[None]],
    *,
    kind: str,
    concurrency: int = SYNC_CONCURRENCY,
) -> None:
    """Run a per-task `update_fn` across every Label Studio task in
    `project_id`, with bounded concurrency and per-task heartbeat.

    Incremental sync via the api-side `LabelStudioSyncCursor`: the
    helper fetches the cursor for `(kind, project_id)`, skips tasks
    whose `updated_at <= cursor.last_synced_at`, and on full success
    advances the cursor to the highest `task.updated_at` it saw. A
    partial failure (TaskGroup all-or-nothing) leaves the cursor where
    it was so the next run reprocesses the same range; per-task PUTs
    are upserts, so replay is safe.

    A missing LS project is logged and treated as a no-op so a single
    rogue project_id doesn't fail the whole sync.
    """
    ls = get_ls_client()

    activity.logger.info(
        "sync_label_studio_project starting kind=%s project_id=%d",
        kind,
        project_id,
    )

    try:
        _ = await asyncio.to_thread(ls.projects.get, project_id)
    except ApiError as e:
        activity.logger.warning(
            "sync_label_studio_project missing kind=%s project_id=%d error=%s",
            kind,
            project_id,
            e,
        )
        return

    tasks = await _list_ls_tasks_with_heartbeat(ls, project_id)
    sem = asyncio.Semaphore(concurrency)

    async with get_fs_client() as fs:
        cursor = await fs.labels.get_sync_cursor(kind, project_id)
        cursor_ts = cursor.last_synced_at if cursor is not None else None

        eligible_tasks: list[Any] = []
        max_seen: datetime | None = cursor_ts
        for task in tasks:
            ts = _coerce_updated_at(getattr(task, "updated_at", None))
            if cursor_ts is not None and ts is not None and ts <= cursor_ts:
                continue
            eligible_tasks.append(task)
            if ts is not None and (max_seen is None or ts > max_seen):
                max_seen = ts

        if not eligible_tasks:
            activity.logger.info(
                "sync_label_studio_project no new tasks kind=%s project_id=%d "
                "total_tasks=%d cursor=%s",
                kind,
                project_id,
                len(tasks),
                cursor_ts,
            )
            return

        activity.logger.info(
            "sync_label_studio_project running kind=%s project_id=%d "
            "total_tasks=%d eligible_tasks=%d cursor=%s",
            kind,
            project_id,
            len(tasks),
            len(eligible_tasks),
            cursor_ts,
        )

        async def _run(task: Any) -> None:
            async with sem:
                await update_fn(fs, task)
                activity.heartbeat()

        async with ExceptionGroupErrorLogging(activity.logger):
            async with asyncio.TaskGroup() as tg:
                for task in eligible_tasks:
                    if activity.is_cancelled():
                        activity.logger.info(
                            "Activity cancelled, stopping sync for project %d",
                            project_id,
                        )
                        return

                    tg.create_task(_run(task))

        if max_seen is not None and max_seen != cursor_ts:
            new_cursor = LabelStudioSyncCursor(
                id=cursor.id if cursor is not None else None,
                kind=kind,
                label_studio_project_id=project_id,
                last_synced_at=max_seen,
            )
            await fs.labels.put_sync_cursor(kind, project_id, new_cursor)
            activity.logger.info(
                "sync_label_studio_project cursor advanced kind=%s "
                "project_id=%d cursor=%s",
                kind,
                project_id,
                max_seen,
            )

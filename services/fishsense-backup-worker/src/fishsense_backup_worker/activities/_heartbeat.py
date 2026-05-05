"""Heartbeat helpers for backup-worker activities.

The backup activities both wrap a long-running blocking call
(`pg_dump` subprocess, `synology-api` NAS list+delete) inside
`asyncio.to_thread`. Because the work doesn't yield to the event
loop, the activity body can't pump heartbeats inline — instead we
spawn a background pump task that fires `activity.heartbeat()` on a
fixed cadence while the blocking work runs, mirroring the api-worker
pattern in `activities/utils.py::_list_ls_tasks_with_heartbeat`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from temporalio import activity

DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0


@asynccontextmanager
async def heartbeat_pump(
    interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> AsyncIterator[None]:
    """Run `activity.heartbeat()` on a ticker for the duration of the
    `async with` block.

    The pump is cancelled cleanly on exit; it can't outlive the
    activity body and won't suppress exceptions raised inside the
    block.
    """

    async def _pump() -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                activity.heartbeat()
        except asyncio.CancelledError:
            return

    pump = asyncio.create_task(_pump())
    try:
        yield
    finally:
        pump.cancel()
        try:
            await pump
        except asyncio.CancelledError:
            pass

"""Background heartbeat pump for long-running SDK activities.

Extracted from `validate_laser_labels_for_dive_activity`, which was the first
activity to need it and is no longer the only one. The failure it exists for is
specific: httpx applies its `read` timeout per byte-gap rather than to a whole
download, so a slowly-streamed multi-MB response body can keep reading for
minutes without tripping httpx — and well past a workflow's `heartbeat_timeout`.
A single slow await inside the activity body would then fail the activity with
no signal about where it hung.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator

from temporalio import activity

# Comfortably under the callers' `heartbeat_timeout=1m` so a single missed
# pump tick still leaves a safety margin.
HEARTBEAT_INTERVAL_SECONDS = 30.0

__all__ = ["HEARTBEAT_INTERVAL_SECONDS", "heartbeat_pump"]


@contextlib.asynccontextmanager
async def heartbeat_pump(
    interval_seconds: float = HEARTBEAT_INTERVAL_SECONDS,
) -> AsyncIterator[None]:
    """Pump `activity.heartbeat()` on a background task for the duration.

    Explicit per-milestone `activity.heartbeat()` calls in an activity body
    stay worthwhile — they are cheap and they bracket the interesting steps for
    diagnostics. This pump covers the case where one of those awaits is itself
    slow.
    """

    async def _pump() -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                activity.heartbeat()
        except asyncio.CancelledError:
            return

    task = asyncio.create_task(_pump())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

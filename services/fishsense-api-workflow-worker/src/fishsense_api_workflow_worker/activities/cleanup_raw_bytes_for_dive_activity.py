"""Activity to delete a dive's raw `.ORF` bytes from the file-exchange.

Runs on the api-worker AFTER `archive_processed_jpegs_to_nas_activity`
succeeds — frees nginx storage by dropping reproducible-from-NAS raw
inputs. JPEGs intentionally stay because labelers' LS task URLs
reference them; their retention is a separate operational decision
(see `MEMORY.md` / `project_jpeg_retention_policy.md`).

Per-checksum DELETE on `/api/v1/exchange/raw/{checksum}.ORF`. nginx
DAV returns 204 on missing-file too, so this is naturally idempotent
under retries.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.file_exchange import StagingFileExchangeClient

CLEANUP_CONCURRENCY = 8

__all__ = [
    "CLEANUP_CONCURRENCY",
    "CleanupRawBytesResult",
    "cleanup_raw_bytes_for_dive_activity",
]


@dataclass
class CleanupRawBytesResult:
    """Per-dive cleanup summary."""

    deleted: int  # checksums whose DELETE returned 200/204/404


@activity.defn
async def cleanup_raw_bytes_for_dive_activity(
    dive_id: int,
) -> CleanupRawBytesResult:
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []

    activity.logger.info(
        "cleaning up raw bytes dive_id=%d images=%d", dive_id, len(images)
    )

    sem = asyncio.Semaphore(CLEANUP_CONCURRENCY)
    deleted = 0

    async with httpx.AsyncClient(
        base_url=settings.file_exchange.url,
        timeout=httpx.Timeout(60.0),
    ) as http:
        exchange = StagingFileExchangeClient(
            base_url=settings.file_exchange.url, http=http
        )

        async def _delete_one(image) -> None:
            nonlocal deleted
            if not image.checksum:
                activity.heartbeat()
                return

            async with sem:
                ok = await exchange.delete_raw(image.checksum)
                if ok:
                    deleted += 1
                activity.heartbeat()

        async with asyncio.TaskGroup() as tg:
            for image in images:
                tg.create_task(_delete_one(image))

    activity.logger.info(
        "raw cleanup done dive_id=%d deleted=%d", dive_id, deleted
    )
    return CleanupRawBytesResult(deleted=deleted)

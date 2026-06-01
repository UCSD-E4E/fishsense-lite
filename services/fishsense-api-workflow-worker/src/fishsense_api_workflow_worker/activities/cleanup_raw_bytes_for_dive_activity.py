"""Activity to delete a dive's staged raw `.ORF` bytes from the Garage
object store (the scratch copy only).

Runs on the api-worker after the data-worker has produced the processed
JPEGs — frees object-store space by dropping the reproducible-from-NAS
raw inputs. The processed JPEGs intentionally stay in Garage (LS reads
them via presign); their retention is a separate operational decision
(see `MEMORY.md` / `jpeg-retention-open-question`).

NAS safety invariant: this only deletes the Garage `raw/{checksum}.ORF`
*scratch* objects. The NAS source `.ORF` is never touched — there is no
NAS-delete path anywhere on the api-worker.

S3 delete_object is idempotent (deleting an absent key is a success),
so this is naturally safe under retries.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.object_store import open_object_store_client

CLEANUP_CONCURRENCY = 8

__all__ = [
    "CLEANUP_CONCURRENCY",
    "CleanupRawBytesResult",
    "cleanup_raw_bytes_for_dive_activity",
]


@dataclass
class CleanupRawBytesResult:
    """Per-dive cleanup summary."""

    deleted: int  # scratch raw objects deleted from Garage


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

    exchange = open_object_store_client()

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

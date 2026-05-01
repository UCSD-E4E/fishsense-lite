"""Activity to stage all raw `.ORF` bytes for a dive from NAS to the
file-exchange.

Runs on the api-worker (which holds the NAS credentials) ahead of the
data-worker child workflow that consumes
`/api/v1/exchange/raw/{checksum}.ORF`. Idempotent: HEAD-checks the
file-exchange first and skips checksums already staged.

Failure semantics: any per-image failure (NAS missing, HTTP error)
raises and aborts the whole activity. The schedule's `overlap=SKIP`
will not retry within the firing, but the next firing of the parent
schedule retries from scratch — staging is bounded by the dive's
image count and skip-on-present makes the retry cheap for already-
staged work.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.file_exchange import StagingFileExchangeClient
from fishsense_api_workflow_worker.nas import NasDownloadClient

STAGE_CONCURRENCY = 8

__all__ = [
    "STAGE_CONCURRENCY",
    "StageRawBytesResult",
    "stage_raw_bytes_for_dive_activity",
]


@dataclass
class StageRawBytesResult:
    """Per-dive staging summary so the parent workflow can log
    counts without re-querying the file-exchange."""

    staged: int  # newly downloaded + uploaded
    skipped_already_present: int
    no_path: int  # images whose `path` was None — skipped, surfaced in count


def _build_nas_client() -> NasDownloadClient:
    return NasDownloadClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


@activity.defn
async def stage_raw_bytes_for_dive_activity(
    dive_id: int,
) -> StageRawBytesResult:
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []

    activity.logger.info(
        "staging raw bytes dive_id=%d images=%d", dive_id, len(images)
    )

    nas = _build_nas_client()
    sem = asyncio.Semaphore(STAGE_CONCURRENCY)

    staged = 0
    skipped = 0
    no_path = 0

    async with httpx.AsyncClient(
        base_url=settings.file_exchange.url,
        timeout=httpx.Timeout(120.0),
    ) as http:
        exchange = StagingFileExchangeClient(
            base_url=settings.file_exchange.url, http=http
        )

        async def _stage_one(image) -> str:
            nonlocal staged, skipped, no_path
            if not image.path:
                no_path += 1
                activity.heartbeat()
                return "no_path"
            if not image.checksum:
                no_path += 1
                activity.heartbeat()
                return "no_checksum"

            async with sem:
                if await exchange.has_raw(image.checksum):
                    skipped += 1
                    activity.heartbeat()
                    return "skipped"

                with tempfile.TemporaryDirectory() as tmpdir:
                    await asyncio.to_thread(
                        nas.download_to,
                        src_path=image.path,
                        dest_dir=tmpdir,
                    )
                    local_path = Path(tmpdir) / os.path.basename(image.path)
                    data = await asyncio.to_thread(local_path.read_bytes)
                    await exchange.upload_raw(image.checksum, data)
                    staged += 1
                    activity.heartbeat()
                    return "staged"

        async with asyncio.TaskGroup() as tg:
            for image in images:
                tg.create_task(_stage_one(image))

    activity.logger.info(
        "staged dive_id=%d staged=%d skipped=%d no_path=%d",
        dive_id,
        staged,
        skipped,
        no_path,
    )
    return StageRawBytesResult(
        staged=staged,
        skipped_already_present=skipped,
        no_path=no_path,
    )

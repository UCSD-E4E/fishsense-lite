"""Activity to archive processed JPEGs from the file-exchange to NAS.

Runs on the api-worker after the data-worker child workflow has
finished writing JPEGs to `{exchange_folder}/{checksum}.JPG`. The
archive lands at
`{processed_jpegs.nas_root_path}/{nas_workflow}/{dive_id}/{checksum}.JPG`.

Idempotent: skips checksums whose JPEG is already on NAS, and
silently ignores file-exchange entries that don't exist (a partially-
processed dive's missing JPEGs aren't fatal — the next preprocess
run produces them and the next archive picks them up).

Failure semantics: any per-image upload error raises and aborts the
parent. Partial archives are fine because re-runs HEAD-skip what's
already on NAS.
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.file_exchange import StagingFileExchangeClient
from fishsense_api_workflow_worker.nas import NasClient

ARCHIVE_CONCURRENCY = 8

__all__ = [
    "ARCHIVE_CONCURRENCY",
    "ArchiveResult",
    "archive_processed_jpegs_to_nas_activity",
]


@dataclass
class ArchiveResult:
    """Per-dive archive summary so the parent workflow can log
    counts without re-walking the NAS."""

    archived: int
    skipped_already_on_nas: int
    skipped_no_jpeg: int  # file-exchange returned 404 for this checksum


def _build_nas_client() -> NasClient:
    return NasClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


def _nas_dir(nas_workflow: str, dive_id: int) -> str:
    return (
        f"{settings.processed_jpegs.nas_root_path.rstrip('/')}"
        f"/{nas_workflow}/{dive_id}"
    )


@activity.defn
async def archive_processed_jpegs_to_nas_activity(
    dive_id: int, exchange_folder: str, nas_workflow: str
) -> ArchiveResult:
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []

    activity.logger.info(
        "archiving processed JPEGs dive_id=%d folder=%s workflow=%s images=%d",
        dive_id,
        exchange_folder,
        nas_workflow,
        len(images),
    )

    nas = _build_nas_client()
    sem = asyncio.Semaphore(ARCHIVE_CONCURRENCY)
    nas_dir = _nas_dir(nas_workflow, dive_id)

    archived = 0
    skipped_already = 0
    skipped_no_jpeg = 0

    async with httpx.AsyncClient(
        base_url=settings.file_exchange.url,
        timeout=httpx.Timeout(120.0),
    ) as http:
        exchange = StagingFileExchangeClient(
            base_url=settings.file_exchange.url, http=http
        )

        async def _archive_one(image) -> None:
            nonlocal archived, skipped_already, skipped_no_jpeg
            if not image.checksum:
                activity.heartbeat()
                return

            nas_file = f"{nas_dir}/{image.checksum}.JPG"

            async with sem:
                already = await asyncio.to_thread(
                    nas.exists, file_path=nas_file
                )
                if already:
                    skipped_already += 1
                    activity.heartbeat()
                    return

                try:
                    data = await exchange.download_processed_jpeg(
                        exchange_folder, image.checksum
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        skipped_no_jpeg += 1
                        activity.heartbeat()
                        return
                    raise

                with tempfile.TemporaryDirectory() as tmpdir:
                    local_path = Path(tmpdir) / f"{image.checksum}.JPG"
                    await asyncio.to_thread(local_path.write_bytes, data)
                    await asyncio.to_thread(
                        nas.upload,
                        dest_dir=nas_dir,
                        src_file_path=str(local_path),
                    )

                archived += 1
                activity.heartbeat()

        async with asyncio.TaskGroup() as tg:
            for image in images:
                tg.create_task(_archive_one(image))

    activity.logger.info(
        "archived dive_id=%d archived=%d skipped_already_on_nas=%d "
        "skipped_no_jpeg=%d",
        dive_id,
        archived,
        skipped_already,
        skipped_no_jpeg,
    )
    return ArchiveResult(
        archived=archived,
        skipped_already_on_nas=skipped_already,
        skipped_no_jpeg=skipped_no_jpeg,
    )

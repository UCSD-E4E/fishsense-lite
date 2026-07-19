"""Activity to stage all raw `.ORF` bytes for a dive from NAS to the
Garage object store.

Runs on the api-worker (which holds the NAS credentials) ahead of the
data-worker child workflow that reads the staged raw `.ORF` from the
Garage `raw/{checksum}.ORF` scratch key. Idempotent: HEAD-checks the
object store first (HeadObject) and skips checksums already staged.

NAS access here is strictly read-only (download). The NAS stays the
source of truth — nothing in this activity ever deletes from it.

Path resolution: `image.path` in the DB is stored relative to the
lab's data-root share (e.g. `2024.06.20.REEF/08_2023/.../P8290052.ORF`);
this activity prepends `e4e_nas.raw_root_path` (default
`/fishsense_data/REEF/data`) before calling FileStation. Without the
prefix, FileStation's download endpoint fails with a 502 (it surfaces
unresolved paths as Bad Gateway on the download API specifically).

Failure semantics: a transient `TransportError` (502 Bad Gateway from
FileStation's download backend) on a single file is retried with
exponential backoff (`NAS_DOWNLOAD_MAX_ATTEMPTS`); anything still
failing after that — or any non-transient error — raises and aborts
the whole activity. Files are never skipped: a silently-missing raw
would stage the dive incomplete and hide a real NAS/data problem, so a
persistent failure is surfaced, not swallowed. The next firing of the
parent schedule retries from scratch — skip-on-present makes that cheap
for already-staged work. `STAGE_CONCURRENCY` is deliberately low so a
backlog of dives doesn't saturate FileStation's download backend (which
502s under concurrent large-`.ORF` load, and can auto-block a client
that hammers it).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from synology_filestation import TransportError
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.object_store import open_object_store_client
from fishsense_api_workflow_worker.nas import NasDownloadClient

# Lowered from 8 → 3: FileStation's download backend (DSM nginx →
# synoscgi) buckles under many concurrent large-`.ORF` transfers and
# starts returning 502; fewer parallel downloads keep us under that
# threshold and stop the worker from tripping the NAS's auto-block.
STAGE_CONCURRENCY = 3

# Per-file download retry. `TransportError` is what the Synology client
# raises for a 502 Bad Gateway (DSM's nginx couldn't reach its own
# FileStation download backend) — usually transient, so retry with
# backoff instead of failing the whole dive on the first blip. We never
# *skip* a file after exhaustion: a silently-missing raw would yield an
# incomplete dive, so exhaustion re-raises and the activity fails loudly.
NAS_DOWNLOAD_MAX_ATTEMPTS = 4
NAS_DOWNLOAD_RETRY_INITIAL_SECONDS = 2.0
NAS_DOWNLOAD_RETRY_BACKOFF = 2.0

__all__ = [
    "STAGE_CONCURRENCY",
    "NAS_DOWNLOAD_MAX_ATTEMPTS",
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




async def _download_with_retry(nas, *, src_path: str, dest_dir: str) -> None:
    """Download one file, retrying a transient `TransportError` (502 Bad
    Gateway from FileStation's download backend) with exponential
    backoff.

    Re-raises on the final attempt — callers must NOT swallow it. Skipping
    a file that can't be fetched would stage the dive incomplete (missing
    raws → missing preprocessed images) and hide a real NAS/data problem;
    a hard failure surfaces it. The backoff also throttles our request
    rate so we stop hammering an already-struggling NAS.
    """
    delay = NAS_DOWNLOAD_RETRY_INITIAL_SECONDS
    for attempt in range(1, NAS_DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            await asyncio.to_thread(
                nas.download_to, src_path=src_path, dest_dir=dest_dir
            )
            return
        except TransportError as exc:
            if attempt >= NAS_DOWNLOAD_MAX_ATTEMPTS:
                raise
            activity.logger.warning(
                "nas download transient failure src=%s attempt=%d/%d: %s; "
                "backing off %.1fs",
                src_path,
                attempt,
                NAS_DOWNLOAD_MAX_ATTEMPTS,
                exc,
                delay,
            )
            activity.heartbeat()
            await asyncio.sleep(delay)
            delay *= NAS_DOWNLOAD_RETRY_BACKOFF


def _resolve_nas_path(relative_path: str) -> str:
    """Join `e4e_nas.raw_root_path` with the DB's relative `image.path`.

    The DB convention is share-relative (no leading slash); FileStation
    requires absolute paths. If the DB path is already absolute, return
    it unchanged so an operator override (or a future path migration)
    isn't double-prefixed.
    """
    if relative_path.startswith("/"):
        return relative_path
    root = settings.e4e_nas.raw_root_path.rstrip("/")
    return f"{root}/{relative_path.lstrip('/')}"


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

    exchange = open_object_store_client()

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
                src_path = _resolve_nas_path(image.path)
                await _download_with_retry(
                    nas, src_path=src_path, dest_dir=tmpdir
                )
                local_path = Path(tmpdir) / os.path.basename(src_path)
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

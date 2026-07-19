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

Failure semantics: retry/backoff is owned by the **bounded, jittered
Temporal `retry_policy`** on the activity call (`STAGE_RAW_RETRY_POLICY`),
NOT an inner loop — an inner retry under Temporal's outer retry is what
produced the 200×-per-file storm that tripped the NAS auto-block
(krg-infra#501). Transient errors (502 `TransportError`, 407 backend
fail-closed, 402 busy) propagate so Temporal backs off and retries a
bounded number of times, then fails the firing; the hourly schedule owns
re-trying after that. A *permanent* error (Synology 408 "no such file")
is raised as a non-retryable ApplicationError so Temporal doesn't
reschedule a doomed staging. Files are never skipped — a silently-missing
raw would stage the dive incomplete and hide a real NAS/data problem, so
failures surface. `STAGE_CONCURRENCY` is deliberately low so a backlog of
dives doesn't saturate FileStation's shared download backend.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from synology_filestation import DSMError
from temporalio import activity
from temporalio.exceptions import ApplicationError

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.object_store import open_object_store_client
from fishsense_api_workflow_worker.nas import NasDownloadClient

# Lowered from 8 → 3: FileStation's download backend (DSM nginx →
# synoscgi) is a shared per-appliance CGI service that buckles under many
# concurrent large-`.ORF` transfers and starts returning 502. A handful
# of streams is what it's sized for; fewer parallel downloads keep us
# under that threshold and stop the worker tripping the NAS's auto-block.
STAGE_CONCURRENCY = 3

# `type` on the non-retryable ApplicationError we raise for a missing
# file; must match `non_retryable_error_types` in STAGE_RAW_RETRY_POLICY.
NAS_FILE_NOT_FOUND_TYPE = "NasFileNotFound"

# Synology FileStation error codes that are *permanent* — retrying can't
# help, so we fail the dive fast (non-retryable) instead of burning the
# Temporal retry budget. 408 = "No such file or directory". Transient
# codes (502 TransportError, 407 backend-fail-closed, 402 busy) are left
# to propagate so the bounded Temporal retry policy backs off and retries.
# NOTE: this string-parses the DSM code because the client doesn't expose
# it structurally yet — remove once `synology-filestation` classifies
# errors upstream (feedback filed).
_PERMANENT_DSM_CODES = frozenset({408})

__all__ = [
    "STAGE_CONCURRENCY",
    "NAS_FILE_NOT_FOUND_TYPE",
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




def _iter_leaf_exceptions(exc: BaseException):
    """Yield leaf (non-group) exceptions from a possibly-nested
    ExceptionGroup, so a wrapped classification can be recovered."""
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            yield from _iter_leaf_exceptions(sub)
    else:
        yield exc


def _dsm_error_code(exc: BaseException) -> int | None:
    """Best-effort extract the Synology FileStation error code from a
    `DSMError` (whose message is `"Synology API error <code>"`). Interim
    until the client exposes the code structurally."""
    match = re.search(r"error\s+(\d+)", str(exc))
    return int(match.group(1)) if match else None


async def _download_one(nas, *, src_path: str, dest_dir: str) -> None:
    """Download a single file. Retry/backoff is intentionally NOT here —
    the bounded, jittered Temporal `retry_policy` on the activity owns
    that (an inner loop under Temporal's outer retry is what produced the
    200×-per-file storm). We only classify: a *permanent* FileStation
    error (e.g. 408 "no such file") becomes a non-retryable
    ApplicationError so Temporal doesn't reschedule a doomed staging;
    transient errors (502/407/402) propagate and Temporal backs off.

    Never skip a file: a silently-missing raw would stage the dive
    incomplete and hide a real NAS/data problem, so failures surface.
    """
    try:
        await asyncio.to_thread(
            nas.download_to, src_path=src_path, dest_dir=dest_dir
        )
    except DSMError as exc:
        code = _dsm_error_code(exc)
        if code in _PERMANENT_DSM_CODES:
            raise ApplicationError(
                f"NAS file not found (Synology {code}): {src_path}",
                type=NAS_FILE_NOT_FOUND_TYPE,
                non_retryable=True,
            ) from exc
        raise


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
                await _download_one(nas, src_path=src_path, dest_dir=tmpdir)
                local_path = Path(tmpdir) / os.path.basename(src_path)
                data = await asyncio.to_thread(local_path.read_bytes)
                await exchange.upload_raw(image.checksum, data)
                staged += 1
                activity.heartbeat()
                return "staged"

    try:
        async with asyncio.TaskGroup() as tg:
            for image in images:
                tg.create_task(_stage_one(image))
    except BaseExceptionGroup as group:
        # TaskGroup wraps every failure in an ExceptionGroup, which would
        # mask a sub-exception's non_retryable flag from Temporal. Surface
        # a permanent classification (408 -> non-retryable ApplicationError)
        # un-wrapped so Temporal honours it and doesn't reschedule a doomed
        # staging; otherwise re-raise the group (transient -> Temporal
        # retries under the bounded policy).
        for leaf in _iter_leaf_exceptions(group):
            if isinstance(leaf, ApplicationError) and leaf.non_retryable:
                # `leaf` already carries its own `from exc` (the DSMError)
                # cause; chaining it to the group would be wrong.
                raise leaf  # pylint: disable=raise-missing-from
        raise

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

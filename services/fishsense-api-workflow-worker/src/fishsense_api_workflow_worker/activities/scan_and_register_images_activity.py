"""Activity: download a batch of frames, hash them, and create their `Image` rows.

The only ingest step that writes. Listing enumerates and preflight reads 1 MB
headers; this one pulls whole files (~14.5 MB each), so its mistakes are the
expensive ones.

Per frame, in order:

1. **Skip if the path is already registered — before downloading.** Checking
   afterwards would be just as correct and would still move the bytes. The
   dive's existing paths are fetched once per batch rather than per frame.
2. `download_to` the NAS file into a temp dir.
3. **Stream `md5` in 8192-byte chunks.** Byte-identical to hashing the whole
   buffer, and it matches `spider/backend.py:67` exactly — the convention all
   ~131k migrated rows follow, re-verified against the NAS corpus-wide. Getting
   this wrong does not error: duplicate detection would silently report zero
   overlap and every re-ingested frame would land canonical.
4. EXIF tag **0x0132**, stamped UTC, offset deliberately not applied.
5. `images.post`, WITHOUT `is_canonical` — the server computes it. Sending it
   would mark every duplicate frame canonical and destroy the distinction the
   canonical-only pipeline gating depends on.

**Reject, never default.** A frame with no readable timestamp is refused rather
than given a placeholder: stage-1 clustering is pure timestamp arithmetic, so a
fabricated value corrupts it invisibly. The batch records the rejection and
carries on — `finalize_dive` refuses to promote a dive with any rejection, so a
bad frame stops the dive rather than entering it, and reporting all of them
beats aborting on whichever failed first.

**Heartbeat per frame, resume from the recorded index.** A batch is many
whole-file downloads; a retry that restarted from zero would re-pull everything
it had already registered.

Retry/backoff belongs to the Temporal policy on the activity call, never an
inner loop — that is what produced the download storm that tripped the NAS
auto-block. A permanent Synology error (408, no such file) becomes a
non-retryable `ApplicationError`; transient ones propagate. Note this is the
opposite of `verify_dive_checksums_activity`, which treats a missing file as a
finding: ingest is trying to *do* something, so a missing frame is a failure.

NAS access is download-only, with a test tripwire asserting the module contains
no write call.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from synology_filestation import DSMError
from temporalio import activity

from fishsense_api_workflow_worker.activities.nas_errors import (
    raise_if_permanent_dsm_error,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.exif import read_exif
from fishsense_api_workflow_worker.nas import NasDownloadClient
from fishsense_shared.ingest_contracts import RejectedImage

# Matches `spider/backend.py:67`. The chunking is not scoping — it is identical
# to hashing the whole buffer — but it keeps a ~15 MB buffer per frame off the
# heap, which is why spider did it and why we do.
_HASH_CHUNK_BYTES = 8192

# EXIF lives at the front; no need to re-read the whole file to reach it.
_EXIF_HEADER_BYTES = 1024 * 1024

__all__ = ["BatchResult", "scan_and_register_images_activity"]


@dataclass
class BatchResult:
    """What one batch did, for the workflow to accumulate."""

    registered: int = 0
    skipped_existing: int = 0
    rejected: List[RejectedImage] = field(default_factory=list)
    #: MAX of the frames' timestamps — how every existing `Dive.dive_datetime`
    #: was derived, and this is the only step that reads them.
    max_taken_datetime: datetime | None = None


def _build_nas_client() -> NasDownloadClient:
    return NasDownloadClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


def _resolve_nas_path(relative_path: str) -> str:
    """Prepend the share root. The DB stores share-relative paths; FileStation
    needs absolute ones, and surfaces an unresolved path as a 502."""
    if relative_path.startswith("/"):
        return relative_path
    root = settings.e4e_nas.raw_root_path.rstrip("/")
    return f"{root}/{relative_path.lstrip('/')}"


def _file_checksum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for blob in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(blob)
    return digest.hexdigest()


def _taken_datetime(path: Path) -> datetime | None:
    """Naive EXIF 0x0132 stamped UTC — the convention ~131k rows follow. The
    camera's recorded offset is deliberately NOT applied: one consistent offset
    is recoverable later, two conventions mixed in one column are not."""
    with open(path, "rb") as handle:
        header = handle.read(_EXIF_HEADER_BYTES)
    raw = read_exif(header).date_time
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _download(nas, src_path: str, dest_dir: str) -> None:
    """One download, classified. No inner retry — the bounded Temporal policy
    owns backoff."""
    try:
        nas.download_to(src_path=src_path, dest_dir=dest_dir)
    except DSMError as exc:
        raise_if_permanent_dsm_error(exc, context=src_path)
        raise


def _read_frame(nas, stored_path: str) -> tuple[str, datetime | None]:
    """Download to a temp dir and return `(checksum, taken_datetime)`.

    Runs in a worker thread: the hash and the EXIF read are blocking file I/O.
    """
    src_path = _resolve_nas_path(stored_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        _download(nas, src_path, tmpdir)
        local = Path(tmpdir) / os.path.basename(src_path)
        return _file_checksum(local), _taken_datetime(local)


def _resume_index() -> int:
    """Where a retry should pick up.

    Temporal replays the last heartbeat's details on retry; frames before that
    index were already registered, so re-downloading them is pure waste.
    """
    details = activity.info().heartbeat_details
    if not details:
        return 0
    try:
        return int(details[0])
    except (TypeError, ValueError):
        return 0


@activity.defn
async def scan_and_register_images_activity(
    dive_id: int, paths: List[str], camera_id: int
) -> BatchResult:
    from fishsense_api_sdk.models.image import (  # pylint: disable=import-outside-toplevel
        Image,
    )

    result = BatchResult()
    nas = _build_nas_client()
    start = _resume_index()

    async with get_fs_client() as fs:
        # One call, not one per frame: the whole point is to skip BEFORE paying
        # for a download.
        existing = {
            image.path for image in (await fs.images.get(dive_id=dive_id) or [])
        }

        for index, stored_path in enumerate(paths):
            if index < start:
                continue
            activity.heartbeat(index)

            if stored_path in existing:
                result.skipped_existing += 1
                continue

            checksum, taken = await asyncio.to_thread(_read_frame, nas, stored_path)
            if taken is None:
                result.rejected.append(
                    RejectedImage(
                        path=stored_path,
                        reason=(
                            "no readable EXIF timestamp (tag 0x0132 or 0x9003); "
                            "refusing to default one because stage-1 clustering "
                            "is pure timestamp arithmetic"
                        ),
                    )
                )
                continue

            await fs.images.post(
                dive_id,
                Image(
                    id=None,
                    path=stored_path,
                    taken_datetime=taken,
                    checksum=checksum,
                    # Required by the model, stripped from the wire by the SDK
                    # unless `set_canonical=True`. The server decides.
                    is_canonical=False,
                    dive_id=dive_id,
                    camera_id=camera_id,
                ),
            )
            result.registered += 1
            if result.max_taken_datetime is None or taken > result.max_taken_datetime:
                result.max_taken_datetime = taken

    activity.logger.info(
        "scanned dive_id=%d frames=%d registered=%d skipped=%d rejected=%d",
        dive_id,
        len(paths),
        result.registered,
        result.skipped_existing,
        len(result.rejected),
    )
    return result

"""Activity: re-hash a migrated dive's frames against the NAS and report.

Answers "do we trust the existing data?" with a measurement instead of an
argument.

Every ingest convention was recovered by reading the retired spider crawler:
`Image.checksum` is `md5` of the whole file (`backend.py:67`) and
`taken_datetime` is naive EXIF tag 0x0132 stamped UTC (`backend.py:20`). That
derivation is solid and the migration chain is traced end to end — but it
proves what *spider* wrote. It does not prove all ~131k rows came from spider,
and no stored value had ever been compared against the bytes on the NAS.

The gap matters because of its failure mode. If existing checksums were
computed even slightly differently, duplicate detection does not error — it
silently reports **zero overlap**. A re-ingest of an already-present dive would
look entirely new, every frame would land `is_canonical=True`, and the
canonical-only gating would have nothing to gate on.

**Read-only, by construction.** No NAS writes, no SDK writes; a test tripwire
asserts this module's source contains no write call. It runs against prod data
to answer a question and must not be able to change the answer.

**Findings, not failures.** A mismatch — or a row whose file is gone — is one
of the answers being looked for, so the run records it and carries on. Only an
infrastructure error (a NAS that is down rather than a file that is absent)
propagates for Temporal to retry. That is the opposite of
`stage_raw_bytes_for_dive_activity`, where a missing file is a non-retryable
failure, and the difference is deliberate: staging is trying to *do* something.

Whole files are downloaded, unlike preflight's ranged 1 MB read — an MD5 of the
whole file needs the whole file. So a dive is ~500 × ~15 MB, and `limit` exists
to sample: answering "does the convention hold" does not need every frame.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from synology_filestation import DSMError
from temporalio import activity

from fishsense_api_workflow_worker.activities.nas_errors import dsm_error_code
from fishsense_api_workflow_worker.activities.nas_frames import (
    build_nas_client,
)
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.exif import read_exif
from fishsense_shared.ingest_contracts import (
    ChecksumMismatch,
    VerifyChecksumsReport,
)

# Matches `spider/backend.py:67` exactly. The chunking is not scoping — it is
# byte-identical to hashing the whole buffer — but it keeps a 15 MB buffer per
# frame off the heap, which is why spider did it and why we do.
_HASH_CHUNK_BYTES = 8192

# EXIF sits at the front of the file; no need to re-read 15 MB to compare a
# timestamp we already have on disk.
_EXIF_HEADER_BYTES = 1024 * 1024

# "No such file or directory". Here it is a *finding* (the row outlived its
# file), not the permanent failure it is when staging.
_DSM_NOT_FOUND = 408

__all__ = ["verify_dive_checksums_activity"]


def _resolve_nas_path(relative_path: str) -> str:
    """Prepend the share root, as the staging activity does. Absolute paths
    pass through so a hand-corrected row isn't double-prefixed."""
    if relative_path.startswith("/"):
        return relative_path
    root = settings.e4e_nas.raw_root_path.rstrip("/")
    return f"{root}/{relative_path.lstrip('/')}"


def _file_checksum(path: Path) -> str:
    """`md5` of the whole file, streamed. The convention of record."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for blob in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(blob)
    return digest.hexdigest()


def _exif_taken_datetime(path: Path) -> datetime | None:
    """Naive EXIF 0x0132 stamped UTC — agreement with the migrated rows, not
    correctness. The camera's recorded offset is deliberately not applied."""
    with open(path, "rb") as handle:
        header = handle.read(_EXIF_HEADER_BYTES)
    raw = read_exif(header).date_time
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _as_utc(value: datetime | None) -> datetime | None:
    """Compare like with like: a row read back without tzinfo is still UTC by
    the migration's construction, so an aware/naive mix must not read as a
    mismatch."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def _verify_one(nas, image, report: VerifyChecksumsReport) -> None:
    """Download one frame, compare, and record. Never raises on a finding."""
    src_path = _resolve_nas_path(image.path)
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            nas.download_to(src_path=src_path, dest_dir=tmpdir)
        except DSMError as exc:
            if dsm_error_code(exc) == _DSM_NOT_FOUND:
                report.missing_on_nas.append(
                    ChecksumMismatch(
                        image_id=image.id, path=image.path, stored=image.checksum
                    )
                )
                return
            # A NAS that is down, rather than a file that is absent. Propagate
            # so Temporal's bounded policy retries rather than recording a
            # finding that is really an outage.
            raise

        local = Path(tmpdir) / os.path.basename(src_path)
        computed = _file_checksum(local)
        exif_taken = _exif_taken_datetime(local)

    if not image.checksum:
        report.no_stored_checksum.append(
            ChecksumMismatch(image_id=image.id, path=image.path, computed=computed)
        )
    elif image.checksum == computed:
        report.checksum_matched += 1
    else:
        report.mismatches.append(
            ChecksumMismatch(
                image_id=image.id,
                path=image.path,
                stored=image.checksum,
                computed=computed,
            )
        )

    stored_taken = _as_utc(image.taken_datetime)
    if stored_taken != exif_taken:
        report.timestamp_mismatches.append(
            ChecksumMismatch(
                image_id=image.id,
                path=image.path,
                stored=stored_taken.isoformat() if stored_taken else None,
                computed=exif_taken.isoformat() if exif_taken else None,
            )
        )


@activity.defn
async def verify_dive_checksums_activity(
    dive_id: int, limit: int | None = None
) -> VerifyChecksumsReport:
    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []

    # Deliberately NOT filtered to canonical rows. The duplicates are half the
    # table and are exactly the population whose provenance is least certain,
    # so excluding them would skip the rows most worth checking.
    ordered = sorted(images, key=lambda i: i.path or "")
    report = VerifyChecksumsReport(dive_id=dive_id, total_in_dive=len(ordered))
    selected = ordered if limit is None else ordered[:limit]

    nas = build_nas_client()
    for index, image in enumerate(selected):
        if not image.path:
            continue
        # Details carry the index so a retry resumes rather than re-downloading
        # everything already checked — whole files, so that is real bandwidth.
        activity.heartbeat(index)
        await asyncio.to_thread(_verify_one, nas, image, report)
        report.checked += 1

    activity.logger.info(
        "verified dive_id=%d checked=%d matched=%d mismatched=%d "
        "timestamp_mismatched=%d missing=%d no_checksum=%d",
        dive_id,
        report.checked,
        report.checksum_matched,
        len(report.mismatches),
        len(report.timestamp_mismatches),
        len(report.missing_on_nas),
        len(report.no_stored_checksum),
    )
    return report

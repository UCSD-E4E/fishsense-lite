"""Shared NAS-frame plumbing: client construction, path resolution, and the two
conventions every raw frame is read under.

Extracted when `duplicate-code` flagged 42 identical lines between
`scan_and_register_images_activity` and `verify_dive_checksums_activity`, and
`_build_nas_client` turned out to be copied six times. Two of those copies
disagreeing would not fail loudly — the checksum and timestamp conventions here
are exactly the kind that break *silently*:

  * A checksum computed differently from `spider/backend.py:67` does not error.
    Duplicate detection simply reports **zero overlap**, every re-ingested frame
    lands `is_canonical=True`, and the canonical-only pipeline gating has
    nothing to gate on.
  * A timestamp read from a different tag, or with the camera's offset applied,
    does not error either. It silently disagrees with ~131k existing rows and
    corrupts stage-1 clustering, which is pure timestamp arithmetic.

Both are now defined once. Re-verified against the live corpus 2026-08-17:
`VerifyAllDivesChecksumsWorkflow` re-hashed 1,619 frames across all 272 canonical
dives with zero disagreements, so `file_checksum` below is the convention of
record, not an inference from source.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.exif import read_exif
from fishsense_api_workflow_worker.nas import NasDownloadClient

# Matches `spider/backend.py:67` exactly. The chunking is not scoping — it is
# byte-identical to hashing the whole buffer — but it keeps a ~15 MB buffer per
# frame off the heap, which is why spider did it and why we do.
HASH_CHUNK_BYTES = 8192

# EXIF sits at the front of an ORF; no need to re-read ~15 MB to reach it.
EXIF_HEADER_BYTES = 1024 * 1024

__all__ = [
    "EXIF_HEADER_BYTES",
    "HASH_CHUNK_BYTES",
    "build_nas_client",
    "file_checksum",
    "read_taken_datetime",
    "resolve_nas_path",
]


def build_nas_client() -> NasDownloadClient:
    """The api-worker's NAS client, from settings.

    Read at call time rather than import time so a settings change is picked up
    without a code change — and so importing an activity module never touches
    config (Dynaconf validates every validator on first attribute access).
    """
    return NasDownloadClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


def resolve_nas_path(relative_path: str) -> str:
    """Prepend `e4e_nas.raw_root_path` to a share-relative DB path.

    The DB convention is share-relative; FileStation needs absolute. Worth
    getting right because FileStation surfaces an unresolved path as a **502**,
    not a 404, so the failure looks transient. An already-absolute path passes
    through, so a hand-corrected row is not double-prefixed.
    """
    if relative_path.startswith("/"):
        return relative_path
    root = settings.e4e_nas.raw_root_path.rstrip("/")
    return f"{root}/{relative_path.lstrip('/')}"


def file_checksum(path: Path) -> str:
    """`md5` of the whole file, streamed. The convention of record."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for blob in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(blob)
    return digest.hexdigest()


def read_taken_datetime(path: Path) -> datetime | None:
    """Naive EXIF tag 0x0132 stamped UTC, or None if unreadable.

    The camera's recorded offset is deliberately **not** applied: matching the
    ~131k existing rows matters more than being right, because one consistent
    offset is recoverable later and two conventions mixed in one column are not.

    None means "no usable timestamp" and callers must treat it as a rejection
    rather than substituting a default — stage-1 clustering cannot tell a
    fabricated timestamp from a real one.
    """
    with open(path, "rb") as handle:
        header = handle.read(EXIF_HEADER_BYTES)
    raw = read_exif(header).date_time
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None

"""Workflow-input and report DTOs for dive ingestion.

These live here for the same reason `preprocess_contracts` does: they are the
contract between two packages that must not import each other. fishsense-api
starts `IngestDiveWorkflow` and reads its progress; the api-worker runs it.
Neither depends on the other, and both already depend on fishsense-shared.

One request means **one dive**. The images are the `.ORF` files directly inside
the named folder, not a recursive walk — a dive has always been exactly one
directory (the legacy crawler assigned `dive = image.parent`), so recursing
would merge dives that are separate rows in prod today. Subdirectories holding
`.ORF`s are reported so the operator can submit them separately, which is the
Olympus counter-rollover case.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel

__all__ = [
    "ChecksumMismatch",
    "IngestDiveRequest",
    "IngestPreflight",
    "IngestProgress",
    "IngestReport",
    "PreflightImage",
    "RejectedImage",
    "SubfolderReport",
    "VerifyChecksumsReport",
]


class IngestDiveRequest(BaseModel):
    """What an operator submits. One request, one dive."""

    #: NAS path relative to `e4e_nas.raw_root_path`.
    dive_path: str
    #: Defaults to the leaf directory name. `Dive.name` feeds Label Studio
    #: project titles, so it is worth getting right at ingest.
    dive_name: str | None = None

    #: Override camera resolution. When unset the camera is resolved from each
    #: frame's Olympus MakerNote serial; there is deliberately no fallback to
    #: EXIF `Artist`, because guessing a camera silently gives stage 14 the
    #: wrong intrinsics.
    camera_id: int | None = None

    #: HIGH or the hourly cohorts never pick the dive up. Ingest still creates
    #: the dive at LOW and only flips it on success — see `IngestReport`.
    priority: str = "HIGH"

    dive_slate_id: int | None = None

    #: A fish-only dive with no slate frames of its own can never self-calibrate,
    #: so stage 14 can never measure it. That is not detectable from the files,
    #: which is why intent must be stated: exactly one of these is required.
    calibration_dive_id: int | None = None
    self_calibrates: bool = False

    flip_dive_slate: bool = False

    #: Preflight only: list, read EXIF headers, validate, write nothing.
    dry_run: bool = False
    #: Re-hash an already-ingested folder and report mismatches. Never writes.
    verify_existing: bool = False


class PreflightImage(BaseModel):
    """One frame as seen by preflight, before anything is written."""

    path: str
    size: int
    #: Naive EXIF `DateTime` (tag 0x0132), stamped UTC on write to match the
    #: ~111k existing rows. None means the frame will be rejected.
    taken_datetime: datetime | None = None
    #: The camera's own UTC offset, recorded but deliberately NOT applied.
    exif_offset: str | None = None
    serial_number: str | None = None
    artist: str | None = None


class SubfolderReport(BaseModel):
    """A subdirectory holding `.ORF`s — a *separate dive*, not extra frames.

    Surfaced rather than ingested so the operator submits it themselves. This
    is the Olympus rollover case: the TG-6 wraps its frame counter and starts a
    child folder mid-dive.
    """

    path: str
    orf_count: int


class DuplicateOverlap(BaseModel):
    """How much of this folder already exists under another dive.

    Containment is `|new ∩ existing| / |new|` over content checksums — a set
    operation, so it is immune to filenames and ordering and degrades to a
    partial overlap. It replaces the legacy whole-dive MD5 digest, which was
    all-or-nothing and basename-sensitive, and therefore wrong most times it
    was consulted on a corpus that is ~50% duplicates.
    """

    dive_id: int
    shared_images: int
    containment: float


class IngestPreflight(BaseModel):
    """The dry-run result: everything ingest would do, having written nothing."""

    dive_path: str
    images: List[PreflightImage] = []
    subfolders: List[SubfolderReport] = []
    #: Resolved from the MakerNote serial, or the request's override.
    resolved_camera_id: int | None = None
    resolved_camera_name: str | None = None
    total_bytes: int = 0
    #: Every problem at once, not first-wins — an operator should see all of
    #: them in one round trip. Non-empty means the workflow refuses to proceed.
    errors: List[str] = []
    warnings: List[str] = []
    duplicate_overlap: List[DuplicateOverlap] = []


class RejectedImage(BaseModel):
    """A frame ingest declined to write, and why.

    Any rejection blocks the dive's promotion to HIGH: a partially-ingested
    dive must never enter the pipeline.
    """

    path: str
    reason: str


class ChecksumMismatch(BaseModel):
    """One row whose stored value disagrees with the file on the NAS.

    Both values are carried because "some rows disagree" is not actionable —
    the useful finding is *how* they disagree, which is what says whether the
    migration used a different algorithm or the file itself changed.
    """

    image_id: int | None = None
    path: str
    stored: str | None = None
    computed: str | None = None


class VerifyChecksumsReport(BaseModel):
    """Result of re-hashing a migrated dive against the NAS.

    Read-only. Every ingest convention was recovered by reading the retired
    spider crawler's source, which proves what *spider* wrote but not that all
    ~131k rows came from spider — and no stored value had ever been compared
    against the bytes still on the NAS.

    The gap is worth closing because of how it fails: if existing checksums
    were computed differently, duplicate detection does not error, it silently
    reports zero overlap. A re-ingest of an already-present dive would look
    entirely new and every frame would land canonical.
    """

    dive_id: int
    #: Rows in the dive, before any sampling limit.
    total_in_dive: int = 0
    checked: int = 0
    checksum_matched: int = 0
    mismatches: List[ChecksumMismatch] = []
    #: Stored `taken_datetime` disagreeing with EXIF 0x0132 stamped UTC.
    #: Tracked apart from checksums: a wrong checksum breaks duplicate
    #: detection, a wrong timestamp breaks stage-1 clustering.
    timestamp_mismatches: List[ChecksumMismatch] = []
    #: The row exists but its file does not — itself one of the answers.
    missing_on_nas: List[ChecksumMismatch] = []
    #: NULL `checksum` on a migrated row; the column duplicate detection joins
    #: on, so a blank one is a finding rather than a skip.
    no_stored_checksum: List[ChecksumMismatch] = []


class IngestProgress(BaseModel):
    """Shape of the workflow's `progress` query, polled by the portal."""

    state: str = "starting"
    dive_id: int | None = None
    total: int = 0
    scanned: int = 0
    registered: int = 0
    skipped_existing: int = 0
    rejected: int = 0
    current_path: str | None = None


class IngestReport(BaseModel):
    """The workflow's return value."""

    dive_path: str
    dive_id: int | None = None
    total: int = 0
    registered: int = 0
    skipped_existing: int = 0
    rejected: List[RejectedImage] = []
    #: MAX of the frames' timestamps, matching how every existing
    #: `Dive.dive_datetime` was derived.
    dive_datetime: datetime | None = None
    #: True only when every listed frame was persisted. This is what allows the
    #: dive to be flipped to HIGH — priority is the commit flag.
    committed: bool = False
    preflight: IngestPreflight | None = None
    duplicate_overlap: List[DuplicateOverlap] = []

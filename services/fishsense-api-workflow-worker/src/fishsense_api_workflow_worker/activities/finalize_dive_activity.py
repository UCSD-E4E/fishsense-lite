"""Activity: verify the set is complete, then flip the dive to its real priority.

The other half of the two-phase commit. `create_dive_activity` wrote the dive at
`priority=LOW`, which keeps it out of every hourly cohort; this promotes it —
**and only if every listed frame is accounted for.**

Two refusals, both non-retryable:

* **any rejection** — a frame with no readable timestamp was refused rather than
  given a fabricated one, so the dive is missing an image. Promoting it would
  put an incomplete set into stage-1 clustering, which has no way to know.
* **`registered + skipped != total`** — a frame was neither written nor
  recognised as already present. That is worse than a rejection: a rejection is
  reported, a gap is silence.

Non-retryable because neither is transient. Retrying re-reads the same bytes and
reaches the same conclusion; the fix is an operator looking at the report.

It also computes **content overlap** (§4.2.1 layer 2), which has to happen here
rather than in preflight: containment is `|new ∩ existing| / |new|` over content
checksums, and the checksums only exist once the scan has written the rows. Being
a set operation on hashes it is immune to filenames and ordering, and degrades to
a partial overlap — the properties the legacy whole-dive MD5 digest lacked, which
is why it disappointed on a corpus that is ~50% duplicates.

Overlap is **reported, never blocking**. Re-ingesting the same frames under a
second dive path is legitimate and has already happened in prod (dives 64 and 66
are both `082929_FishModels_FSL07`); the duplicate rows land non-canonical and
are invisible to every pipeline cohort.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from temporalio import activity
from temporalio.exceptions import ApplicationError

from fishsense_api_workflow_worker.activities.create_dive_activity import leaf_name
from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_shared.ingest_contracts import (
    DuplicateOverlap,
    IngestDiveRequest,
    IngestReport,
    RejectedImage,
)

# `type` on the non-retryable failure, so a retry policy can name it.
INCOMPLETE_INGEST_TYPE = "IncompleteIngest"

# Look checksums up in chunks: the endpoint caps a request at
# MAX_CHECKSUM_LOOKUP (1000), and a large dive can exceed that.
_LOOKUP_CHUNK = 500

__all__ = ["INCOMPLETE_INGEST_TYPE", "IngestTotals", "finalize_dive_activity"]


@dataclass
class IngestTotals:
    """What the scan batches added up to, accumulated by the workflow."""

    total: int = 0
    registered: int = 0
    skipped_existing: int = 0
    rejected: List[RejectedImage] = field(default_factory=list)
    max_taken_datetime: datetime | None = None


def _chunks(items: List[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _content_overlap(fs, dive_id: int) -> List[DuplicateOverlap]:
    """How much of this dive's content already exists under other dives."""
    images = await fs.images.get(dive_id=dive_id) or []
    checksums = sorted({i.checksum for i in images if i.checksum})
    if not checksums:
        return []

    shared: dict[int, set[str]] = defaultdict(set)
    for chunk in _chunks(checksums, _LOOKUP_CHUNK):
        for checksum, hits in (await fs.images.lookup_checksums(chunk)).items():
            for hit in hits:
                other = hit.get("dive_id")
                # Skip our own rows: they are in the lookup by construction, and
                # counting them would report containment 1.0 against ourselves
                # on every single ingest.
                if other is not None and other != dive_id:
                    shared[other].add(checksum)

    return [
        DuplicateOverlap(
            dive_id=other,
            shared_images=len(found),
            containment=len(found) / len(checksums),
        )
        for other, found in sorted(shared.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]


@activity.defn
async def finalize_dive_activity(
    dive_id: int, request: IngestDiveRequest, totals: IngestTotals
) -> IngestReport:
    from fishsense_api_sdk.models.dive import Dive
    from fishsense_api_sdk.models.priority import Priority

    accounted = totals.registered + totals.skipped_existing
    if totals.rejected:
        raise ApplicationError(
            f"refusing to promote dive {dive_id}: {len(totals.rejected)} frame(s) "
            f"rejected — {'; '.join(r.reason for r in totals.rejected[:3])}. "
            "The dive stays at LOW so no pipeline stage picks up a partial set.",
            type=INCOMPLETE_INGEST_TYPE,
            non_retryable=True,
        )
    if accounted != totals.total:
        raise ApplicationError(
            f"refusing to promote dive {dive_id}: {accounted} of {totals.total} "
            "frames accounted for. A frame was neither written nor recognised as "
            "already present, which is silence rather than a reported failure.",
            type=INCOMPLETE_INGEST_TYPE,
            non_retryable=True,
        )

    async with get_fs_client() as fs:
        overlap = await _content_overlap(fs, dive_id)

        await fs.dives.post(
            Dive(
                id=dive_id,
                name=request.dive_name or leaf_name(request.dive_path),
                path=request.dive_path,
                # The scan read every frame in full; preflight only saw a 1 MB
                # prefix. MAX is how every existing `Dive.dive_datetime` was
                # derived.
                dive_datetime=totals.max_taken_datetime,
                # THE COMMIT FLAG.
                priority=Priority(request.priority),
                flip_dive_slate=request.flip_dive_slate,
                camera_id=None,
                dive_slate_id=request.dive_slate_id,
                calibration_dive_id=request.calibration_dive_id,
            )
        )

    for item in overlap:
        activity.logger.warning(
            "dive %d shares %d/%d frames with dive %d (containment %.2f); "
            "those images are non-canonical",
            dive_id,
            item.shared_images,
            totals.total,
            item.dive_id,
            item.containment,
        )
    activity.logger.info(
        "committed dive %d at %s: registered=%d skipped=%d",
        dive_id,
        request.priority,
        totals.registered,
        totals.skipped_existing,
    )
    return IngestReport(
        dive_path=request.dive_path,
        dive_id=dive_id,
        total=totals.total,
        registered=totals.registered,
        skipped_existing=totals.skipped_existing,
        rejected=[],
        dive_datetime=totals.max_taken_datetime,
        committed=True,
        duplicate_overlap=overlap,
    )

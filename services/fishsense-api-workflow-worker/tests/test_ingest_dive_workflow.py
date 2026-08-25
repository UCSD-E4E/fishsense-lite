"""Contract tests for `IngestDiveWorkflow` — the end-to-end ingest.

In-process Temporal worker with every activity stubbed. What is pinned here is
the *protocol*, not the activities' behaviour:

  * a dry run and a failed preflight both leave without writing anything;
  * the dive is created before any frame is registered, and promoted only after;
  * frames are batched, and the batches' counts add up into one report.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.finalize_dive_activity import (
    IngestTotals,
)
from fishsense_api_workflow_worker.activities.list_dive_folder_activity import (
    DiveFolderListing,
)
from fishsense_api_workflow_worker.activities.scan_and_register_images_activity import (  # noqa: E501  pylint: disable=line-too-long
    BatchResult,
)
from fishsense_api_workflow_worker.workflows.ingest_dive_workflow import (
    BATCH_SIZE,
    IngestDiveWorkflow,
)
from fishsense_shared.ingest_contracts import (
    IngestDiveRequest,
    IngestPreflight,
    IngestReport,
    PreflightImage,
)

TASK_QUEUE = "test-ingest"
FOLDER = "2024.06.20.REEF/082929_FishModels_FSL07"
T = datetime(2024, 8, 21, 9, 30, 0, tzinfo=timezone.utc)


def _activities(calls, *, frames=2, errors=(), rejected=()):
    from fishsense_api_workflow_worker.nas import NasEntry

    images = [
        PreflightImage(path=f"{FOLDER}/{i:04d}.ORF", size=1, taken_datetime=T)
        for i in range(frames)
    ]

    @activity.defn(name="list_dive_folder_activity")
    async def _list(request: IngestDiveRequest) -> DiveFolderListing:
        calls.append("list")
        return DiveFolderListing(
            folder_path=f"/root/{request.dive_path}",
            files=[
                NasEntry(path=i.path, name=i.path[-8:], is_dir=False, size=1)
                for i in images
            ],
        )

    @activity.defn(name="preflight_ingest_activity")
    async def _preflight(  # pylint: disable=unused-argument
        request: IngestDiveRequest, listing: DiveFolderListing
    ) -> IngestPreflight:
        calls.append("preflight")
        return IngestPreflight(
            dive_path=listing.folder_path,
            images=images,
            resolved_camera_id=7,
            errors=list(errors),
        )

    @activity.defn(name="create_dive_activity")
    async def _create(  # pylint: disable=unused-argument
        request: IngestDiveRequest, preflight: IngestPreflight
    ) -> int:
        calls.append("create")
        return 412

    @activity.defn(name="scan_and_register_images_activity")
    async def _scan(  # pylint: disable=unused-argument
        dive_id: int, paths: list, camera_id: int
    ) -> BatchResult:
        calls.append(f"scan:{len(paths)}")
        return BatchResult(
            registered=len(paths),
            rejected=list(rejected),
            max_taken_datetime=T,
        )

    @activity.defn(name="finalize_dive_activity")
    async def _finalize(
        dive_id: int, request: IngestDiveRequest, totals: IngestTotals
    ) -> IngestReport:
        calls.append("finalize")
        return IngestReport(
            dive_path=request.dive_path,
            dive_id=dive_id,
            total=totals.total,
            registered=totals.registered,
            skipped_existing=totals.skipped_existing,
            dive_datetime=totals.max_taken_datetime,
            committed=True,
        )

    return [_list, _preflight, _create, _scan, _finalize]


async def _run(request, **kwargs):
    calls: list[str] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[IngestDiveWorkflow],
            activities=_activities(calls, **kwargs),
        ):
            report = await env.client.execute_workflow(
                IngestDiveWorkflow.run,
                request,
                id=f"{TASK_QUEUE}-{uuid.uuid4()}",
                task_queue=TASK_QUEUE,
            )
    return report, calls


def _request(**kwargs):
    kwargs.setdefault("dive_path", FOLDER)
    kwargs.setdefault("self_calibrates", True)
    return IngestDiveRequest(**kwargs)


# ── the happy path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_creates_the_dive_before_registering_and_finalizes_after():
    """The two-phase commit, as an ordering. `create` must precede every scan
    (frames need a dive to belong to) and `finalize` must follow all of them —
    it is the step that opens the commit flag."""
    report, calls = await _run(_request())

    assert calls[:3] == ["list", "preflight", "create"]
    assert calls[-1] == "finalize"
    assert report.committed is True
    assert report.dive_id == 412


@pytest.mark.asyncio
async def test_frames_are_scanned_in_batches():
    """A failure should cost one batch of ~14.5 MB downloads, not a whole dive."""
    frames = BATCH_SIZE * 2 + 3
    _report, calls = await _run(_request(), frames=frames)

    scans = [c for c in calls if c.startswith("scan:")]
    assert scans == [f"scan:{BATCH_SIZE}", f"scan:{BATCH_SIZE}", "scan:3"]


@pytest.mark.asyncio
async def test_batch_counts_accumulate_into_one_report():
    frames = BATCH_SIZE + 1
    report, _calls = await _run(_request(), frames=frames)

    assert report.total == frames
    assert report.registered == frames
    assert report.dive_datetime == T


# ── writing nothing ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_dry_run_writes_nothing_and_returns_the_preflight():
    report, calls = await _run(_request(dry_run=True))

    assert calls == ["list", "preflight"]
    assert report.committed is False
    assert report.preflight is not None
    assert report.dive_id is None


@pytest.mark.asyncio
async def test_a_failed_preflight_writes_nothing():
    """Preflight reports every fault at once; the workflow's job is simply not
    to proceed. No dive, no images — the operator fixes and resubmits."""
    report, calls = await _run(
        _request(), errors=["camera 7 has no intrinsics"]
    )

    assert calls == ["list", "preflight"]
    assert report.committed is False
    assert report.preflight.errors == ["camera 7 has no intrinsics"]


# ── the report carries the preflight ──────────────────────────────────


@pytest.mark.asyncio
async def test_the_committed_report_still_carries_its_preflight():
    """Warnings — a leaf-name collision, a subfolder that is really another
    dive, an Artist disagreeing with the resolved camera — live in the
    preflight. Dropping it on success would discard everything the operator was
    meant to see."""
    report, _calls = await _run(_request())

    assert report.preflight is not None
    assert report.preflight.resolved_camera_id == 7

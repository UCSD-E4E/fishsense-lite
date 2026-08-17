"""Contract tests for VerifyAllDivesChecksumsWorkflow — the migration audit sweep.

All ~479 dives came through the migration, so the question "does the recovered
convention actually hold?" is asked of the whole corpus, not one dive.

It runs on the api-worker inside the slot, on the NAS's own network — never
driven from a workstation, because ~930 GB does not belong on a WAN link and
the NAS credentials live in the slot. Temporal is what lets the run be started
from anywhere while the bytes stay on the LAN.

Cost shapes the rest: 65,981 canonical images at ~14.5 MB each is **~930 GB**.
Individual transfers are already paced by the client (SMB-preferred, throttled,
FileStation fallback), so the constraint is contention — that ~930 GB shares one
NAS with the hourly staging activities doing real pipeline work. So:

  * `limit_per_dive` is a runtime argument. Whether a dive came through a
    *different ingest path* is a per-dive property, so ~5 frames answers it at
    ~2% of the cost; every frame answers the different question "is any single
    row corrupt". Same workflow serves both.
  * Dives are verified **one at a time**. The sweep is diagnostic and must
    never outrank real pipeline work for NAS bandwidth.
  * One dive's failure must not abandon the sweep — losing hours of transfer
    because dive 300 of 479 was unreachable would be its own outage.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.verify_all_dives_checksums_workflow import (  # noqa: E501  pylint: disable=line-too-long
    VerifyAllDivesChecksumsWorkflow,
)
from fishsense_shared.ingest_contracts import (
    ChecksumMismatch,
    VerifyChecksumsReport,
)

TASK_QUEUE = "test-verify-sweep"


def _activities(dive_ids, *, per_dive=None, fail_on=()):
    """Stub the selector and the per-dive verifier.

    `per_dive` maps dive_id -> report; `fail_on` names dives whose activity
    raises, standing in for a dive whose NAS reads exhausted their retries.
    """
    calls: list[tuple[int, int | None]] = []

    @activity.defn(name="select_canonical_dive_ids_activity")
    async def select() -> list[int]:
        return list(dive_ids)

    @activity.defn(name="verify_dive_checksums_activity")
    async def verify(dive_id: int, limit: int | None) -> VerifyChecksumsReport:
        calls.append((dive_id, limit))
        if dive_id in fail_on:
            raise RuntimeError(f"NAS unreachable for dive {dive_id}")
        if per_dive and dive_id in per_dive:
            return per_dive[dive_id]
        return VerifyChecksumsReport(
            dive_id=dive_id, total_in_dive=55, checked=5, checksum_matched=5
        )

    return [select, verify], calls


async def _run(dive_ids, *args, per_dive=None, fail_on=(), **kwargs):
    acts, calls = _activities(dive_ids, per_dive=per_dive, fail_on=fail_on)
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[VerifyAllDivesChecksumsWorkflow],
            activities=acts,
        ):
            report = await env.client.execute_workflow(
                VerifyAllDivesChecksumsWorkflow.run,
                args=args or (None,),
                id=f"{TASK_QUEUE}-{uuid.uuid4()}",
                task_queue=TASK_QUEUE,
                **kwargs,
            )
    return report, calls


# ── the sweep ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_visits_every_canonical_dive_the_selector_returns():
    report, calls = await _run([11, 22, 33], 5)

    assert [c[0] for c in calls] == [11, 22, 33]
    assert report.dives_requested == 3
    assert report.dives_verified == 3
    assert report.images_checked == 15


@pytest.mark.asyncio
async def test_the_per_dive_limit_is_forwarded_to_each_dive():
    """The knob that decides whether this is a 20 GB sample or a 930 GB full
    sweep. Nothing else about the workflow changes between the two."""
    _, calls = await _run([11, 22], 5)

    assert calls == [(11, 5), (22, 5)]


@pytest.mark.asyncio
async def test_no_limit_means_every_frame_of_every_dive():
    _, calls = await _run([11], None)

    assert calls == [(11, None)]


@pytest.mark.asyncio
async def test_dives_are_verified_one_at_a_time():
    """Serial by design. The client already paces individual transfers; this is
    the layer above it. Verification is diagnostic and shares one NAS with the
    hourly staging activities doing real pipeline work, so a parallel sweep
    would starve them for days."""
    report, calls = await _run([11, 22, 33, 44], 5)

    # A concurrent fan-out could interleave; strict input order proves serial.
    assert [c[0] for c in calls] == [11, 22, 33, 44]
    assert report.dives_verified == 4


# ── findings ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_findings_are_carried_per_dive_and_clean_dives_are_kept():
    """Clean rows are the result too. Dropping them would make "verified, fine"
    indistinguishable from "never reached", which is the whole point of an
    audit."""
    dirty = VerifyChecksumsReport(
        dive_id=22,
        total_in_dive=55,
        checked=5,
        checksum_matched=4,
        mismatches=[
            ChecksumMismatch(
                image_id=9, path="a/b.ORF", stored="0" * 32, computed="1" * 32
            )
        ],
    )
    report, _ = await _run([11, 22], 5, per_dive={22: dirty})

    assert len(report.dives) == 2
    assert [d.dive_id for d in report.dives_with_findings] == [22]
    assert report.dives_with_findings[0].mismatches[0].stored == "0" * 32
    assert report.checksum_matched == 9


@pytest.mark.asyncio
async def test_a_dive_that_cannot_be_verified_is_recorded_and_the_sweep_goes_on():
    """A full sweep is days of transfer. Abandoning it because one dive was
    unreachable would be its own outage — and an unreachable dive must not read
    as clean, which is why `error` is a separate field rather than an absence
    of findings."""
    report, calls = await _run([11, 22, 33], 5, fail_on=(22,))

    # Distinct ids, in order: dive 22 appears several times because the retry
    # policy legitimately re-attempts it, and the point of the test is that 33
    # was still reached afterwards — not how many attempts 22 got.
    attempted = list(dict.fromkeys(c[0] for c in calls))
    assert attempted == [11, 22, 33]
    assert report.dives_verified == 2
    errored = [d for d in report.dives if d.error]
    assert [d.dive_id for d in errored] == [22]
    assert errored[0].is_clean is False


@pytest.mark.asyncio
async def test_an_explicit_dive_list_overrides_the_selector():
    """For re-running just the dives a sample flagged, without re-walking the
    whole corpus."""
    _, calls = await _run([11, 22, 33], 5, [33])

    assert [c[0] for c in calls] == [33]


# ── observability ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_progress_is_queryable_while_the_sweep_runs():
    """A 930 GB sweep runs for days. "Is it still going, and how far?" has to be
    answerable without waiting for the return value."""
    acts, _ = _activities([11, 22])
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[VerifyAllDivesChecksumsWorkflow],
            activities=acts,
        ):
            handle = await env.client.start_workflow(
                VerifyAllDivesChecksumsWorkflow.run,
                args=(5,),
                id=f"{TASK_QUEUE}-{uuid.uuid4()}",
                task_queue=TASK_QUEUE,
            )
            await handle.result()
            # Query by method reference, not by name string: the reference
            # carries the return type, so the payload decodes to the model.
            # `handle.query("progress")` hands back a bare dict.
            progress = await handle.query(
                VerifyAllDivesChecksumsWorkflow.progress
            )

    assert progress.state == "done"
    assert progress.total_dives == 2
    assert progress.dives_done == 2
    assert progress.images_checked == 10

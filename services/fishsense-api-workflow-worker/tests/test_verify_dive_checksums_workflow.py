"""Workflow contract test for VerifyDiveChecksumsWorkflow.

In-process Temporal worker with the activity stubbed. Pins the two things an
operator depends on when they run this by hand: both arguments reach the
activity, and the sampling limit is genuinely optional — a dive is ~500 frames
at ~15 MB, so `--input <dive_id>` alone must mean "check everything", not
"crash because the second argument is missing".
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.verify_dive_checksums_workflow import (  # noqa: E501  pylint: disable=line-too-long
    VerifyDiveChecksumsWorkflow,
)
from fishsense_shared.ingest_contracts import VerifyChecksumsReport

TASK_QUEUE = "test-verify-checksums"


def _stub(seen: list):
    @activity.defn(name="verify_dive_checksums_activity")
    async def stub_activity(
        dive_id: int, limit: int | None
    ) -> VerifyChecksumsReport:
        seen.append((dive_id, limit))
        return VerifyChecksumsReport(
            dive_id=dive_id,
            total_in_dive=55,
            checked=limit or 55,
            checksum_matched=limit or 55,
        )

    return stub_activity


async def _run(*args):
    """Execute the workflow against a stubbed activity, returning
    `(report, calls)` so a test can assert on both."""
    seen: list[tuple[int, int | None]] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[VerifyDiveChecksumsWorkflow],
            activities=[_stub(seen)],
        ):
            report = await env.client.execute_workflow(
                VerifyDiveChecksumsWorkflow.run,
                args=args,
                id=f"{TASK_QUEUE}-{uuid.uuid4()}",
                task_queue=TASK_QUEUE,
            )
    return report, seen


@pytest.mark.asyncio
async def test_forwards_dive_id_and_limit_and_returns_the_report():
    report, seen = await _run(412, 25)

    assert seen == [(412, 25)]
    assert report.dive_id == 412
    assert report.checked == 25
    assert report.total_in_dive == 55


@pytest.mark.asyncio
async def test_the_limit_is_optional_and_defaults_to_the_whole_dive():
    """`temporal workflow start --input 412` with no second argument is the
    obvious way to run this, and it must mean "check every frame"."""
    report, seen = await _run(412)

    assert seen == [(412, None)]
    assert report.checked == 55

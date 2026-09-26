"""Workflow contract for `RemediateLaserSupersedesWorkflow`.

Dry run is the default and must never reach the apply activity. Apply needs
both the flag and the digest of the plan it recomputes to equal the reviewed
report's; a plan with nothing left to revive is a successful no-op, which is
what makes re-applying safe.
"""

from __future__ import annotations

import uuid

import pytest
from fishsense_shared.laser_remediation import (
    RemediateLaserSupersedesInput,
    RemediationDiveRequest,
    revival_digest,
)
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.remediate_laser_supersedes_workflow import (  # noqa: E501  pylint: disable=line-too-long
    RemediateLaserSupersedesWorkflow,
)

pytestmark = pytest.mark.asyncio

QUEUE = "test-remediate-laser"
# What each dive's plan would revive, as the stub activities see it.
PLANS = {7: [10, 21], 8: [], 9: [5]}


def _row(dive_id, revive_ids, *, excluded=False):
    return {
        "dive_id": dive_id,
        "status": "excluded" if excluded else "flagged",
        "positives": 60,
        "superseded_now": len(revive_ids) + 1,
        "superseded_after": 1 if not excluded else len(revive_ids) + 1,
        "revive_ids": [] if excluded else revive_ids,
        "excluded_kept": [],
        "unjudged_superseded": 0,
        "reflection_suspect": None,
        "revive_on_calibration_frames": [],
        "revive_on_images_with_another_live_label": [],
    }


async def _run(request, plans=None):
    plans = PLANS if plans is None else plans
    planned, applied = [], []

    @activity.defn(name="plan_laser_supersede_remediation_activity")
    async def plan_stub(req: RemediationDiveRequest) -> dict:
        planned.append(req)
        return _row(req.dive_id, plans[req.dive_id], excluded=req.dive_excluded)

    @activity.defn(name="apply_laser_supersede_remediation_activity")
    async def apply_stub(req: RemediationDiveRequest) -> int:
        applied.append(req)
        return len(req.revive_ids)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=QUEUE,
            workflows=[RemediateLaserSupersedesWorkflow],
            activities=[plan_stub, apply_stub],
        ):
            report = await env.client.execute_workflow(
                RemediateLaserSupersedesWorkflow.run,
                request,
                id=f"test-remediate-{uuid.uuid4()}",
                task_queue=QUEUE,
            )
    return report, planned, applied


def _digest(plans=None):
    return revival_digest((plans or PLANS).items())


async def test_a_dry_run_plans_every_dive_and_writes_nothing():
    report, planned, applied = await _run(
        RemediateLaserSupersedesInput(dive_ids=[7, 8, 9])
    )

    assert sorted(r.dive_id for r in planned) == [7, 8, 9]
    assert applied == []
    assert report["mode"] == "dry_run"
    assert report["plan_sha256"] == _digest()
    assert report["totals"]["to_revive"] == 3
    assert [row["dive_id"] for row in report["dives"]] == [7, 8, 9]


async def test_exclusions_reach_the_plan():
    report, planned, _ = await _run(
        RemediateLaserSupersedesInput(
            dive_ids=[7, 9], excluded_dive_ids=[9], excluded_label_ids=[21]
        )
    )

    by_dive = {r.dive_id: r for r in planned}
    assert by_dive[9].dive_excluded is True
    assert by_dive[7].excluded_label_ids == [21]
    assert report["excluded_dive_ids"] == [9]
    assert report["excluded_label_ids"] == [21]


async def test_apply_without_a_matching_digest_writes_nothing():
    with pytest.raises(WorkflowFailureError):
        await _run(
            RemediateLaserSupersedesInput(
                dive_ids=[7, 9], apply=True, expected_plan_sha256="0" * 64
            )
        )


async def test_the_apply_flag_alone_is_not_enough():
    with pytest.raises(WorkflowFailureError):
        await _run(RemediateLaserSupersedesInput(dive_ids=[7, 9], apply=True))


async def test_apply_with_the_reviewed_digest_writes_exactly_the_plan():
    plans = {7: [10, 21], 9: [5]}
    report, _, applied = await _run(
        RemediateLaserSupersedesInput(
            dive_ids=[7, 9], apply=True, expected_plan_sha256=_digest(plans)
        ),
        plans=plans,
    )

    assert {r.dive_id: r.revive_ids for r in applied} == {7: [10, 21], 9: [5]}
    assert report["mode"] == "apply"
    assert report["applied"] == {"7": 2, "9": 1}


async def test_re_applying_after_success_is_a_no_op():
    """After the first apply the plan is empty, so the old digest no longer
    matches — and that must be a clean nothing-to-do, not a failure."""
    reviewed = _digest({7: [10]})

    report, _, applied = await _run(
        RemediateLaserSupersedesInput(
            dive_ids=[7], apply=True, expected_plan_sha256=reviewed
        ),
        plans={7: []},
    )

    assert applied == []
    assert report["mode"] == "apply_noop"

# pylint: disable=unused-argument
"""Workflow contract test for EvaluateLaserAutoAcceptParentWorkflow.

The backlog drain. Pins:
  1. Empty cohort -> returns None, wakes nothing, dispatches nothing.
  2. A selected dive -> child on the data-worker **CPU** queue with the
     deterministic id `auto-accept-laser-{dive_id}`, then the apply step.
  3. A dive the gate cleared nothing on -> no apply step.

(2)'s queue matters and its failure is silent: the stub child is registered on
`DATA_PROCESSING_TASK_QUEUE` only, so a regression sending it to the GPU queue
hangs until the execution timeout rather than failing an assertion.

(3) is not just an optimisation. Every dive in this backlog already has Label
Studio tasks, so the apply step walks the whole project; doing that for a dive
the gate refused is a pointless sweep over exactly the dive whose predictions
could not be trusted.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_shared import LaserAutoAcceptSummary
from fishsense_api_workflow_worker.workflows._dispatch import (
    DATA_PROCESSING_TASK_QUEUE,
)
from fishsense_api_workflow_worker.workflows.evaluate_laser_auto_accept_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    EvaluateLaserAutoAcceptParentWorkflow,
)

# Dive id the stub gate refuses — stands for a dive whose predictions did not
# agree. Keyed on the id so the stub stays deterministic under the sandbox.
_REFUSED_DIVE_ID = 99


@workflow.defn(name="EvaluateLaserAutoAcceptWorkflow")
class _StubGate:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, dive_id: int) -> LaserAutoAcceptSummary:
        await workflow.execute_activity(
            "_record_gate_dispatch",
            args=(workflow.info().workflow_id, dive_id),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        refused = dive_id == _REFUSED_DIVE_ID
        return LaserAutoAcceptSummary(
            dive_id=dive_id,
            eligible=not refused,
            reason="weak_consensus" if refused else None,
            n_points=30,
            auto_accepted=0 if refused else 7,
            verdicts=(
                {"dive_ineligible": 30} if refused else {"auto_accepted": 7}
            ),
        )


def _stubs(dive_id, gate_ids, applied, woken):
    @activity.defn(name="select_next_high_priority_dive_for_laser_auto_accept_activity")
    async def selector() -> int | None:
        return dive_id

    @activity.defn(name="ensure_data_worker_running_activity")
    async def ensure_cpu() -> None:
        woken.append(True)

    @activity.defn(name="_record_gate_dispatch")
    async def record_gate(workflow_id: str, d: int) -> None:
        gate_ids.append(workflow_id)

    @activity.defn(name="apply_laser_auto_accept_for_dive_activity")
    async def apply_auto_accept(d: int) -> int:
        applied.append(d)
        return len(applied)

    return [selector, ensure_cpu, record_gate, apply_auto_accept]


async def _run(env, dive_id, gate_ids, applied, woken):
    activities = _stubs(dive_id, gate_ids, applied, woken)
    async with Worker(
        env.client,
        task_queue="test-auto-accept-parent",
        workflows=[EvaluateLaserAutoAcceptParentWorkflow],
        activities=activities,
    ), Worker(
        env.client,
        task_queue=DATA_PROCESSING_TASK_QUEUE,
        workflows=[_StubGate],
        activities=activities,
    ):
        return await env.client.execute_workflow(
            EvaluateLaserAutoAcceptParentWorkflow.run,
            id=f"auto-accept-parent-{uuid.uuid4()}",
            task_queue="test-auto-accept-parent",
        )


@pytest.mark.asyncio
async def test_empty_backlog_returns_none_and_touches_nothing():
    """The steady state once the backlog has drained. It must not wake the
    data-worker to discover there is no work — the sweeper would then be
    fighting this schedule for the replica count every hour."""
    gate_ids: List[str] = []
    applied: List[int] = []
    woken: List[bool] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, None, gate_ids, applied, woken)
    assert result is None
    assert not gate_ids
    assert not applied
    assert not woken


@pytest.mark.asyncio
async def test_selected_dive_is_judged_then_applied():
    gate_ids: List[str] = []
    applied: List[int] = []
    woken: List[bool] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, 442, gate_ids, applied, woken)
    assert result == 442
    assert woken
    assert gate_ids == ["auto-accept-laser-442"]
    assert applied == [442]


@pytest.mark.asyncio
async def test_a_refused_dive_is_judged_but_not_applied():
    gate_ids: List[str] = []
    applied: List[int] = []
    woken: List[bool] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, _REFUSED_DIVE_ID, gate_ids, applied, woken)
    assert result == _REFUSED_DIVE_ID
    assert gate_ids == [f"auto-accept-laser-{_REFUSED_DIVE_ID}"]
    assert not applied


@pytest.mark.asyncio
async def test_child_id_matches_the_predict_parent_so_a_dive_is_never_judged_twice():
    """Both paths into the gate use `auto-accept-laser-{dive_id}`. With
    ALLOW_DUPLICATE that is not a lock, but it does mean a manual run and a
    scheduled one collide loudly rather than silently double-judging, and it
    keeps a dive's gate runs findable under one id in Temporal."""
    gate_ids: List[str] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        await _run(env, 7, gate_ids, [], [])
    assert gate_ids == ["auto-accept-laser-7"]

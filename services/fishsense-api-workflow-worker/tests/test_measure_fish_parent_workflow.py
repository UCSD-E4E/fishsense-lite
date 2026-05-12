# pylint: disable=unused-argument
"""Workflow contract test for MeasureFishParentWorkflow.

Pins down:
  1. Selector returns None -> parent returns None, no child dispatch.
  2. Selector returns an id -> child dispatched on the data-worker task
     queue with a deterministic id (`measure-fish-{dive_id}`) and the
     dive_id payload, parent returns the dive_id.

The child stub forwards its workflow_id + payload through a recording
activity so the contract test can observe what the parent dispatched
across the workflow sandbox boundary.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.measure_fish_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DATA_PROCESSING_TASK_QUEUE,
    MeasureFishParentWorkflow,
)


@workflow.defn(name="MeasureFishWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    """Stand-in for the data-worker's MeasureFishWorkflow."""

    @workflow.run
    async def run(self, dive_id: int) -> dict:
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(workflow.info().workflow_id, dive_id),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        return {
            "measured": 0,
            "dropped_nan": 0,
            "missing_laser_or_headtail": 0,
            "missing_cluster": 0,
        }


def _make_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(workflow_id: str, dive_id: int) -> None:
        captures.append((workflow_id, dive_id))

    return record_child_dispatch


@activity.defn(name="ensure_data_worker_running_activity")
async def _stub_ensure_data_worker_running() -> int:
    return 0


def _make_stub_selector(selector_result: Optional[int]):
    selector_calls: List[None] = []

    @activity.defn(
        name="select_next_high_priority_dive_for_measure_fish_activity"
    )
    async def stub_select() -> Optional[int]:
        selector_calls.append(None)
        return selector_result

    return stub_select, selector_calls


@pytest.mark.asyncio
async def test_dispatches_child_with_deterministic_id_and_dive_payload():
    stub_select, selector_calls = _make_stub_selector(440)
    child_runs: List[tuple] = []
    record = _make_recording_activity(child_runs)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage14-parent",
            workflows=[MeasureFishParentWorkflow],
            activities=[stub_select, _stub_ensure_data_worker_running],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[record],
        ):
            result = await env.client.execute_workflow(
                MeasureFishParentWorkflow.run,
                id=f"test-stage14-parent-{uuid.uuid4()}",
                task_queue="test-stage14-parent",
            )

    assert result == 440
    assert len(selector_calls) == 1
    assert len(child_runs) == 1
    child_id, child_dive_id = child_runs[0]
    assert child_id == "measure-fish-440"
    assert child_dive_id == 440


@pytest.mark.asyncio
async def test_returns_none_when_selector_finds_no_dive():
    stub_select, selector_calls = _make_stub_selector(None)
    child_runs: List[tuple] = []
    record = _make_recording_activity(child_runs)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage14-parent-empty",
            workflows=[MeasureFishParentWorkflow],
            activities=[stub_select, _stub_ensure_data_worker_running],
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[record],
        ):
            result = await env.client.execute_workflow(
                MeasureFishParentWorkflow.run,
                id=f"test-stage14-parent-empty-{uuid.uuid4()}",
                task_queue="test-stage14-parent-empty",
            )

    assert result is None
    assert len(selector_calls) == 1
    assert not child_runs

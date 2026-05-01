"""Workflow contract test for MeasureFishWorkflow."""

from __future__ import annotations

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.measure_fish_activity import (
    MeasureFishResult,
)
from fishsense_data_processing_workflow_worker.workflows.measure_fish_workflow import (
    MeasureFishWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_invokes_activity_with_dive_id_and_returns_result():
    calls: List[int] = []
    expected = MeasureFishResult(
        measured=3, dropped_nan=1, missing_laser_or_headtail=2, missing_cluster=0
    )

    @activity.defn(name="measure_fish_activity")
    async def stub_activity(dive_id: int) -> MeasureFishResult:
        calls.append(dive_id)
        return expected

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage14",
            workflows=[MeasureFishWorkflow],
            activities=[stub_activity],
        ):
            result = await env.client.execute_workflow(
                MeasureFishWorkflow.run,
                383,
                id=f"test-stage14-{uuid.uuid4()}",
                task_queue="test-stage14",
            )

    assert calls == [383]
    assert result == expected

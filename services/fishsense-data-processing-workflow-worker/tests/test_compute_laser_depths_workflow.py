"""Workflow contract test for ComputeLaserDepthsWorkflow."""

from __future__ import annotations

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.compute_laser_depths_activity import (  # noqa: E501  pylint: disable=line-too-long
    ComputeLaserDepthsResult,
)
from fishsense_data_processing_workflow_worker.workflows.compute_laser_depths_workflow import (  # noqa: E501  pylint: disable=line-too-long
    ComputeLaserDepthsWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_invokes_activity_with_dive_id_and_returns_result():
    calls: List[int] = []
    expected = ComputeLaserDepthsResult(
        computed=4, skipped_current=2, skipped_unusable_label=1
    )

    @activity.defn(name="compute_laser_depths_activity")
    async def stub_activity(dive_id: int) -> ComputeLaserDepthsResult:
        calls.append(dive_id)
        return expected

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-laser-depth",
            workflows=[ComputeLaserDepthsWorkflow],
            activities=[stub_activity],
        ):
            result = await env.client.execute_workflow(
                ComputeLaserDepthsWorkflow.run,
                383,
                id=f"test-laser-depth-{uuid.uuid4()}",
                task_queue="test-laser-depth",
            )

    assert calls == [383]
    assert result == expected

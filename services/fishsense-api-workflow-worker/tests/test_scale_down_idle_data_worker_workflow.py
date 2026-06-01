"""Workflow contract test for ScaleDownIdleDataWorkerWorkflow.

It's a thin wrapper — just pin that it delegates to
``scale_down_data_worker_if_idle_activity`` and passes the result
through.
"""

from __future__ import annotations

import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.scale_down_idle_data_worker_workflow import (  # noqa: E501  pylint: disable=line-too-long
    ScaleDownIdleDataWorkerWorkflow,
)


@pytest.mark.parametrize("scaled_down", [True, False])
@pytest.mark.asyncio
async def test_workflow_returns_activity_result(scaled_down: bool):
    calls: list = []

    @activity.defn(name="scale_down_data_worker_if_idle_activity")
    async def stub_scale_down() -> bool:
        calls.append(True)
        return scaled_down

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-data-worker-sweeper",
            workflows=[ScaleDownIdleDataWorkerWorkflow],
            activities=[stub_scale_down],
        ):
            result = await env.client.execute_workflow(
                ScaleDownIdleDataWorkerWorkflow.run,
                id=f"test-data-worker-sweeper-{uuid.uuid4()}",
                task_queue="test-data-worker-sweeper",
            )

    assert result is scaled_down
    assert calls == [True]

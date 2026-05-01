"""Workflow contract test for PerformLaserCalibrationWorkflow.

Drives the workflow end-to-end against an in-process Temporal test server
(`WorkflowEnvironment.start_time_skipping()`). The activity is replaced
with a stub that records its dive_id so we can assert the workflow
forwards the input + propagates the activity's return value unchanged.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.perform_laser_calibration_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PerformLaserCalibrationWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_invokes_activity_with_dive_id_and_returns_its_result():
    calls: List[int] = []

    @activity.defn(name="perform_laser_calibration_activity")
    async def stub_activity(dive_id: int) -> int | None:
        calls.append(dive_id)
        return 42

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage13",
            workflows=[PerformLaserCalibrationWorkflow],
            activities=[stub_activity],
        ):
            result = await env.client.execute_workflow(
                PerformLaserCalibrationWorkflow.run,
                427,
                id=f"test-stage13-{uuid.uuid4()}",
                task_queue="test-stage13",
            )

    assert calls == [427]
    assert result == 42


@pytest.mark.asyncio
async def test_workflow_propagates_none_from_activity_when_no_slate():
    @activity.defn(name="perform_laser_calibration_activity")
    async def stub_activity(_: int) -> int | None:
        return None

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage13-none",
            workflows=[PerformLaserCalibrationWorkflow],
            activities=[stub_activity],
        ):
            result = await env.client.execute_workflow(
                PerformLaserCalibrationWorkflow.run,
                999,
                id=f"test-stage13-none-{uuid.uuid4()}",
                task_queue="test-stage13-none",
            )

    assert result is None

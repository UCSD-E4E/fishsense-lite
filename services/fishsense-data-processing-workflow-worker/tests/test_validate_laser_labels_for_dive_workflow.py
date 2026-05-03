"""Workflow contract test for ValidateLaserLabelsForDiveWorkflow.

Pins that the workflow forwards `dive_id` to the activity and returns
its result unchanged — mirrors the shape of
test_perform_laser_calibration_workflow.py.
"""

from __future__ import annotations

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.validate_laser_labels_for_dive_workflow import (  # noqa: E501  pylint: disable=line-too-long
    ValidateLaserLabelsForDiveWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_invokes_activity_and_returns_outlier_count():
    calls: List[int] = []

    @activity.defn(name="validate_laser_labels_for_dive_activity")
    async def stub_activity(dive_id: int) -> int:
        calls.append(dive_id)
        return 3

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-validate-laser",
            workflows=[ValidateLaserLabelsForDiveWorkflow],
            activities=[stub_activity],
        ):
            result = await env.client.execute_workflow(
                ValidateLaserLabelsForDiveWorkflow.run,
                123,
                id=f"test-validate-laser-{uuid.uuid4()}",
                task_queue="test-validate-laser",
            )

    assert calls == [123]
    assert result == 3

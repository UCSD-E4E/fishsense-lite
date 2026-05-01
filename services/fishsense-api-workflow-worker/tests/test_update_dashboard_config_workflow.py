"""Workflow contract test for UpdateDashboardConfigWorkflow.

Asserts the workflow first fetches the four project lists in one
activity, then forwards them as a 4-tuple to the writer activity.
"""

from __future__ import annotations

import uuid
from typing import Any, List, Tuple

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.update_dashboard_config_workflow import (
    UpdateDashboardConfigWorkflow,
)


@pytest.mark.asyncio
async def test_workflow_forwards_four_project_lists_to_writer():
    writer_args: List[Tuple[Any, Any, Any, Any]] = []

    @activity.defn(name="get_label_studio_projects_activity")
    async def stub_get_projects():
        return ([1], [2, 3], [4], [5, 6, 7])

    @activity.defn(name="write_dashboard_config_activity")
    async def stub_write(laser, species, headtail, slate) -> None:
        writer_args.append((laser, species, headtail, slate))

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-dashboard-config",
            workflows=[UpdateDashboardConfigWorkflow],
            activities=[stub_get_projects, stub_write],
        ):
            await env.client.execute_workflow(
                UpdateDashboardConfigWorkflow.run,
                id=f"test-dashboard-config-{uuid.uuid4()}",
                task_queue="test-dashboard-config",
            )

    assert writer_args == [([1], [2, 3], [4], [5, 6, 7])]

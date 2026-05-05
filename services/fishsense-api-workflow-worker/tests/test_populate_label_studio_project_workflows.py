"""Workflow contract tests for the four populate-LS-project workflows.

Each workflow does the same thing: idempotently create-or-get a
per-dive LS project (via `create_<stage>_label_studio_project_activity`,
which builds a `"{dive.name} - <Stage>"` title), then push tasks for
that dive to that one project.

These tests cover:
  * Create runs first with the dive id, returns the per-dive project,
    and populate runs once with `(dive_id, project_id)`.
  * The populate count is propagated unchanged (including 0 for a
    no-op dive whose images are already labeled).
"""

from __future__ import annotations

import uuid
from typing import List, Type

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.populate_dive_slate_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateDiveSlateLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_headtail_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateHeadTailLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_laser_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateLaserLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_species_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateSpeciesLabelStudioProjectWorkflow,
)


_STAGES = [
    (
        PopulateLaserLabelStudioProjectWorkflow,
        "create_laser_label_studio_project_activity",
        "populate_laser_label_studio_project_activity",
    ),
    (
        PopulateSpeciesLabelStudioProjectWorkflow,
        "create_species_label_studio_project_activity",
        "populate_species_label_studio_project_activity",
    ),
    (
        PopulateHeadTailLabelStudioProjectWorkflow,
        "create_headtail_label_studio_project_activity",
        "populate_headtail_label_studio_project_activity",
    ),
    (
        PopulateDiveSlateLabelStudioProjectWorkflow,
        "create_dive_slate_label_studio_project_activity",
        "populate_dive_slate_label_studio_project_activity",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_cls,create_name,populate_name", _STAGES)
async def test_workflow_creates_per_dive_project_then_populates(
    workflow_cls: Type,
    create_name: str,
    populate_name: str,
):
    """Create activity is called with the dive id and its return value
    is fanned in to populate as the project id for the same dive."""
    create_calls: List[int] = []
    populate_calls: List[tuple] = []

    @activity.defn(name=create_name)
    async def stub_create(dive_id: int) -> int:
        create_calls.append(dive_id)
        return 555

    @activity.defn(name=populate_name)
    async def stub_populate(dive_id: int, project_id: int) -> int:
        populate_calls.append((dive_id, project_id))
        return 7

    queue = f"test-{workflow_cls.__name__}-per-dive"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_create, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    assert result == 7
    assert create_calls == [427]
    assert populate_calls == [(427, 555)]


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_cls,create_name,populate_name", _STAGES)
async def test_workflow_returns_zero_when_populate_has_no_work(
    workflow_cls: Type,
    create_name: str,
    populate_name: str,
):
    """Populate returns 0 (every image already labeled). The workflow
    propagates 0 without erroring."""

    @activity.defn(name=create_name)
    async def stub_create(_dive_id: int) -> int:
        return 555

    @activity.defn(name=populate_name)
    async def stub_populate(_dive_id: int, _project_id: int) -> int:
        return 0

    queue = f"test-{workflow_cls.__name__}-noop"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_create, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    assert result == 0

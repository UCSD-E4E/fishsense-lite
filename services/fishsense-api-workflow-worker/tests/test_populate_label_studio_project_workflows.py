"""Workflow contract tests for the four populate-LS-project workflows.

Each workflow does the same thing: query SQL for the set of LS
project IDs that are actively being labeled for this stage, then fan
out the per-project populate activity with bounded concurrency.

These tests cover three invariants per workflow:
  * Empty project list short-circuits to 0 with no populate calls.
  * Non-empty list invokes the per-project activity exactly once per ID.
  * Returned counts sum across projects.

A separate concurrency test guards the workflow-level
`PROJECT_CONCURRENCY` semaphore — Phase 1 of the sync workflow had an
unbounded fan-out regression and we don't want to repeat it here.
"""

from __future__ import annotations

import asyncio
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
        "get_active_laser_label_studio_project_ids_activity",
        "populate_laser_label_studio_project_activity",
    ),
    (
        PopulateSpeciesLabelStudioProjectWorkflow,
        "get_active_species_label_studio_project_ids_activity",
        "populate_species_label_studio_project_activity",
    ),
    (
        PopulateHeadTailLabelStudioProjectWorkflow,
        "get_active_headtail_label_studio_project_ids_activity",
        "populate_headtail_label_studio_project_activity",
    ),
    (
        PopulateDiveSlateLabelStudioProjectWorkflow,
        "get_active_dive_slate_label_studio_project_ids_activity",
        "populate_dive_slate_label_studio_project_activity",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_cls,get_active_name,populate_name", _STAGES)
async def test_workflow_fans_out_one_populate_per_active_project(
    workflow_cls: Type, get_active_name: str, populate_name: str
):
    populate_calls: List[tuple] = []

    @activity.defn(name=get_active_name)
    async def stub_get_active() -> List[int]:
        return [101, 202, 303]

    @activity.defn(name=populate_name)
    async def stub_populate(dive_id: int, project_id: int) -> int:
        populate_calls.append((dive_id, project_id))
        return 5

    queue = f"test-{workflow_cls.__name__}-fanout"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_get_active, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    assert result == 15  # 3 projects * 5 each
    assert {p for _, p in populate_calls} == {101, 202, 303}
    assert {d for d, _ in populate_calls} == {427}


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_cls,get_active_name,populate_name", _STAGES)
async def test_workflow_short_circuits_when_no_active_projects(
    workflow_cls: Type, get_active_name: str, populate_name: str
):
    populate_called = False

    @activity.defn(name=get_active_name)
    async def stub_get_active() -> List[int]:
        return []

    @activity.defn(name=populate_name)
    async def stub_populate(_dive_id: int, _project_id: int) -> int:
        nonlocal populate_called
        populate_called = True
        return 0

    queue = f"test-{workflow_cls.__name__}-empty"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_get_active, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    assert result == 0
    assert populate_called is False


@pytest.mark.asyncio
async def test_workflow_caps_per_project_concurrency():
    """Phase 1 regression guard: with N >> PROJECT_CONCURRENCY, the
    workflow must serialize the per-project activity through a
    Semaphore(4)."""
    in_flight = 0
    peak_in_flight = 0
    started: List[int] = []
    release_gate = asyncio.Event()

    @activity.defn(name="get_active_laser_label_studio_project_ids_activity")
    async def stub_get_active() -> List[int]:
        return list(range(20))

    @activity.defn(name="populate_laser_label_studio_project_activity")
    async def stub_populate(_dive_id: int, project_id: int) -> int:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        started.append(project_id)
        try:
            await asyncio.wait_for(release_gate.wait(), timeout=5.0)
        finally:
            in_flight -= 1
        return 0

    queue = f"test-populate-laser-conc-{uuid.uuid4()}"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[PopulateLaserLabelStudioProjectWorkflow],
            activities=[stub_get_active, stub_populate],
            max_concurrent_activities=50,
        ):
            wf_task = asyncio.create_task(
                env.client.execute_workflow(
                    PopulateLaserLabelStudioProjectWorkflow.run,
                    1,
                    id=f"{queue}-wf",
                    task_queue=queue,
                )
            )
            await asyncio.sleep(0.5)
            release_gate.set()
            await wf_task

    assert peak_in_flight <= 4, f"peak {peak_in_flight}, expected <= 4"
    assert len(started) == 20

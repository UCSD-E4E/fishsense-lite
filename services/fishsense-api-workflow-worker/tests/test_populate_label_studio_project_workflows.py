"""Workflow contract tests for the four populate-LS-project workflows.

Each workflow does the same thing: idempotently create-or-get the
canonical LS project, query SQL for any additional projects already
holding incomplete labels of this kind, union the two sets, then fan
out the per-project populate activity with bounded concurrency.

These tests cover:
  * Create returns project id N, discovery returns []  -> fan-out
    runs once for {N}. (Bootstrap: a brand-new deployment with no
    legacy project still gets populated.)
  * Create returns N, discovery returns [N, M, ...]    -> fan-out
    deduplicates {N} and runs once for each unique id.
  * Create returns N, discovery returns []             -> when
    populate returns 0 for the new project (no images), the workflow
    still returns 0 cleanly.
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
        "create_laser_label_studio_project_activity",
        "get_active_laser_label_studio_project_ids_activity",
        "populate_laser_label_studio_project_activity",
    ),
    (
        PopulateSpeciesLabelStudioProjectWorkflow,
        "create_species_label_studio_project_activity",
        "get_active_species_label_studio_project_ids_activity",
        "populate_species_label_studio_project_activity",
    ),
    (
        PopulateHeadTailLabelStudioProjectWorkflow,
        "create_headtail_label_studio_project_activity",
        "get_active_headtail_label_studio_project_ids_activity",
        "populate_headtail_label_studio_project_activity",
    ),
    (
        PopulateDiveSlateLabelStudioProjectWorkflow,
        "create_dive_slate_label_studio_project_activity",
        "get_active_dive_slate_label_studio_project_ids_activity",
        "populate_dive_slate_label_studio_project_activity",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_cls,create_name,get_active_name,populate_name", _STAGES
)
async def test_workflow_unions_canonical_and_discovered_projects(
    workflow_cls: Type,
    create_name: str,
    get_active_name: str,
    populate_name: str,
):
    """Canonical (Create) project ID is unioned with the discovery
    query result. Duplicates are deduplicated so populate doesn't
    fan out twice for the same project."""
    populate_calls: List[tuple] = []

    @activity.defn(name=create_name)
    async def stub_create() -> int:
        return 101

    @activity.defn(name=get_active_name)
    async def stub_get_active() -> List[int]:
        return [101, 202, 303]  # 101 overlaps with canonical -> dedupe

    @activity.defn(name=populate_name)
    async def stub_populate(dive_id: int, project_id: int) -> int:
        populate_calls.append((dive_id, project_id))
        return 5

    queue = f"test-{workflow_cls.__name__}-union"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_create, stub_get_active, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    # 3 unique projects (101 deduped) * 5 each.
    assert result == 15
    assert {p for _, p in populate_calls} == {101, 202, 303}
    assert {d for d, _ in populate_calls} == {427}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_cls,create_name,get_active_name,populate_name", _STAGES
)
async def test_workflow_self_bootstraps_when_discovery_empty(
    workflow_cls: Type,
    create_name: str,
    get_active_name: str,
    populate_name: str,
):
    """Discovery returns []; Create returns a single canonical id ->
    populate must still run for that one project. This is the
    bootstrap case for a brand-new deployment (no legacy LS project,
    fresh project from Create has zero incomplete labels yet)."""
    populate_calls: List[tuple] = []

    @activity.defn(name=create_name)
    async def stub_create() -> int:
        return 555

    @activity.defn(name=get_active_name)
    async def stub_get_active() -> List[int]:
        return []

    @activity.defn(name=populate_name)
    async def stub_populate(dive_id: int, project_id: int) -> int:
        populate_calls.append((dive_id, project_id))
        return 3

    queue = f"test-{workflow_cls.__name__}-bootstrap"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_create, stub_get_active, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    assert result == 3
    assert populate_calls == [(427, 555)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_cls,create_name,get_active_name,populate_name", _STAGES
)
async def test_workflow_returns_zero_when_canonical_has_no_work(
    workflow_cls: Type,
    create_name: str,
    get_active_name: str,
    populate_name: str,
):
    """Canonical project exists but the dive has nothing to push
    (e.g. every image is already labeled). Populate runs and returns
    0; the workflow propagates 0 without erroring."""

    @activity.defn(name=create_name)
    async def stub_create() -> int:
        return 555

    @activity.defn(name=get_active_name)
    async def stub_get_active() -> List[int]:
        return []

    @activity.defn(name=populate_name)
    async def stub_populate(_dive_id: int, _project_id: int) -> int:
        return 0

    queue = f"test-{workflow_cls.__name__}-noop"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[workflow_cls],
            activities=[stub_create, stub_get_active, stub_populate],
        ):
            result = await env.client.execute_workflow(
                workflow_cls.run,
                427,
                id=f"{queue}-{uuid.uuid4()}",
                task_queue=queue,
            )

    assert result == 0


@pytest.mark.asyncio
async def test_workflow_caps_per_project_concurrency():
    """Phase 1 regression guard: with N >> PROJECT_CONCURRENCY, the
    workflow must serialize the per-project activity through a
    Semaphore(4)."""
    in_flight = 0
    peak_in_flight = 0
    started: List[int] = []
    release_gate = asyncio.Event()

    @activity.defn(name="create_laser_label_studio_project_activity")
    async def stub_create() -> int:
        # Canonical id distinct from the discovered range so the
        # union has 21 unique projects to fan out.
        return 9999

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
            activities=[stub_create, stub_get_active, stub_populate],
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
    assert len(started) == 21  # 20 discovered + 1 canonical, all unique

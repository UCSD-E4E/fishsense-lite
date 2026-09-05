# pylint: disable=unused-argument
"""Workflow contract test for PopulateSpeciesLabelStudioProjectParentWorkflow."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.populate_species_label_studio_project_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PopulateSpeciesLabelStudioProjectParentWorkflow,
)


@workflow.defn(name="PopulateSpeciesLabelStudioProjectWorkflow")
class _StubPopulateWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, dive_id: int) -> int:
        await workflow.execute_activity(
            "_record_populate",
            args=(workflow.info().workflow_id, dive_id),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        return 0


def _make_selector(dive_ids: List[int], calls: List[int]):
    @activity.defn(name="select_dives_needing_species_population_activity")
    async def select() -> List[int]:
        calls.append(1)
        return dive_ids

    return select


def _make_recorder(captures: List[tuple]):
    @activity.defn(name="_record_populate")
    async def record(workflow_id: str, dive_id: int) -> None:
        captures.append((workflow_id, dive_id))

    return record


@pytest.mark.asyncio
async def test_fans_out_idempotent_populate_child_per_dive():
    dispatched: List[tuple] = []
    selector_calls: List[int] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[
                PopulateSpeciesLabelStudioProjectParentWorkflow,
                _StubPopulateWorkflow,
            ],
            activities=[
                _make_selector([10, 20, 30], selector_calls),
                _make_recorder(dispatched),
            ],
        ):
            result = await env.client.execute_workflow(
                PopulateSpeciesLabelStudioProjectParentWorkflow.run,
                id=f"parent-{uuid.uuid4()}",
                task_queue="test-queue",
            )

    assert result == [10, 20, 30]
    assert len(selector_calls) == 1
    assert {d for _, d in dispatched} == {10, 20, 30}
    assert {wid for wid, _ in dispatched} == {
        "populate-species-10",
        "populate-species-20",
        "populate-species-30",
    }


@pytest.mark.asyncio
async def test_no_dispatch_when_cohort_empty():
    dispatched: List[tuple] = []
    selector_calls: List[int] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-queue",
            workflows=[
                PopulateSpeciesLabelStudioProjectParentWorkflow,
                _StubPopulateWorkflow,
            ],
            activities=[
                _make_selector([], selector_calls),
                _make_recorder(dispatched),
            ],
        ):
            result = await env.client.execute_workflow(
                PopulateSpeciesLabelStudioProjectParentWorkflow.run,
                id=f"parent-{uuid.uuid4()}",
                task_queue="test-queue",
            )

    assert result == []
    assert len(selector_calls) == 1
    assert not dispatched

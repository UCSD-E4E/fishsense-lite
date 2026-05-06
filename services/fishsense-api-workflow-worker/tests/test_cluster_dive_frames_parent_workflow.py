# pylint: disable=unused-argument
"""Workflow contract test for ClusterDiveFramesParentWorkflow."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.cluster_dive_frames_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DATA_PROCESSING_TASK_QUEUE,
    ClusterDiveFramesParentWorkflow,
)
from fishsense_shared import ClusterDiveFrameImage, ClusterDiveFramesInput


_BASE = datetime(2026, 5, 5, 10, 0, 0, tzinfo=timezone.utc)


@workflow.defn(name="DiveFrameClusteringWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: ClusterDiveFramesInput) -> List[List[int]]:
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(
                workflow.info().workflow_id,
                payload.dive_id,
                [img.image_id for img in payload.images],
            ),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        # Return two synthetic clusters so persist gets non-trivial input.
        return [[1, 2], [3, 4]]


def _make_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(
        workflow_id: str, dive_id: int, image_ids: List[int]
    ) -> None:
        captures.append((workflow_id, dive_id, image_ids))

    return record_child_dispatch


def _make_stubs(
    selector_result: Optional[int],
    resolver_result: Optional[ClusterDiveFramesInput],
    persist_calls: List[tuple],
):
    @activity.defn(name="select_next_high_priority_dive_for_clustering_activity")
    async def stub_select() -> Optional[int]:
        return selector_result

    @activity.defn(name="resolve_dive_frame_clustering_inputs_activity")
    async def stub_resolve(dive_id: int) -> ClusterDiveFramesInput:
        assert resolver_result is not None
        return resolver_result

    @activity.defn(name="persist_dive_frame_clusters_activity")
    async def stub_persist(dive_id: int, clusters: List[List[int]]) -> int:
        persist_calls.append((dive_id, clusters))
        return len(clusters)

    return [stub_select, stub_resolve, stub_persist]


@pytest.mark.asyncio
async def test_dispatches_child_with_deterministic_id_and_persists_clusters():
    inputs = ClusterDiveFramesInput(
        dive_id=440,
        images=[
            ClusterDiveFrameImage(image_id=1, taken_datetime=_BASE),
            ClusterDiveFrameImage(
                image_id=2, taken_datetime=_BASE + timedelta(seconds=1)
            ),
        ],
    )
    persist_calls: List[tuple] = []
    activities = _make_stubs(440, inputs, persist_calls)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage1-parent",
            workflows=[ClusterDiveFramesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                ClusterDiveFramesParentWorkflow.run,
                id=f"test-stage1-parent-{uuid.uuid4()}",
                task_queue="test-stage1-parent",
            )

    assert result == 440
    assert len(child_runs) == 1
    child_id, child_dive_id, image_ids = child_runs[0]
    assert child_id == "cluster-440"
    assert child_dive_id == 440
    assert image_ids == [1, 2]
    assert persist_calls == [(440, [[1, 2], [3, 4]])]


@pytest.mark.asyncio
async def test_returns_none_when_no_dive():
    persist_calls: List[tuple] = []
    activities = _make_stubs(None, None, persist_calls)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage1-parent-none",
            workflows=[ClusterDiveFramesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                ClusterDiveFramesParentWorkflow.run,
                id=f"test-stage1-parent-none-{uuid.uuid4()}",
                task_queue="test-stage1-parent-none",
            )

    assert result is None
    assert not child_runs
    assert not persist_calls


@pytest.mark.asyncio
async def test_skips_child_and_persist_when_no_images():
    inputs = ClusterDiveFramesInput(dive_id=440, images=[])
    persist_calls: List[tuple] = []
    activities = _make_stubs(440, inputs, persist_calls)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage1-parent-empty",
            workflows=[ClusterDiveFramesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                ClusterDiveFramesParentWorkflow.run,
                id=f"test-stage1-parent-empty-{uuid.uuid4()}",
                task_queue="test-stage1-parent-empty",
            )

    assert result == 440
    assert not child_runs
    assert not persist_calls

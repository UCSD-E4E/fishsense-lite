# pylint: disable=unused-argument
"""Workflow contract test for PreprocessDiveImagesParentWorkflow."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.preprocess_dive_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DATA_PROCESSING_TASK_QUEUE,
    PreprocessDiveImagesParentWorkflow,
)
from fishsense_shared import PreprocessDiveImagesInput


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@workflow.defn(name="PreprocessDiveImagesWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PreprocessDiveImagesInput) -> None:
        flat = [c for cluster in payload.clusters for c in cluster]
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(workflow.info().workflow_id, payload.dive_id, flat),
            schedule_to_close_timeout=timedelta(seconds=5),
        )


def _make_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(
        workflow_id: str, dive_id: int, checksums: List[str]
    ) -> None:
        captures.append((workflow_id, dive_id, checksums))

    return record_child_dispatch


def _make_stubs(
    selector_result: Optional[int],
    resolver_result: Optional[PreprocessDiveImagesInput],
):
    selector_calls: List[None] = []
    resolver_calls: List[int] = []

    @activity.defn(name="select_next_high_priority_dive_for_dive_image_preprocessing_activity")
    async def stub_select() -> Optional[int]:
        selector_calls.append(None)
        return selector_result

    @activity.defn(name="resolve_dive_image_preprocess_inputs_activity")
    async def stub_resolve(dive_id: int) -> PreprocessDiveImagesInput:
        resolver_calls.append(dive_id)
        assert resolver_result is not None
        return resolver_result

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stub_stage(dive_id: int) -> None:
        return None

    @activity.defn(name="archive_processed_jpegs_to_nas_activity")
    async def stub_archive(
        dive_id: int, exchange_folder: str, nas_workflow: str
    ) -> None:
        return None

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def stub_cleanup(dive_id: int) -> None:
        return None

    return (
        [
            stub_select,
            stub_resolve,
            stub_stage,
            stub_archive,
            stub_cleanup,
        ],
        selector_calls,
        resolver_calls,
    )


@pytest.mark.asyncio
async def test_dispatches_child_with_deterministic_id_and_clusters():
    inputs = PreprocessDiveImagesInput(
        dive_id=440,
        clusters=[["a", "b"], ["c"]],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, selector_calls, resolver_calls = _make_stubs(440, inputs)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-parent",
            workflows=[PreprocessDiveImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessDiveImagesParentWorkflow.run,
                id=f"test-stage2-parent-{uuid.uuid4()}",
                task_queue="test-stage2-parent",
            )

    assert result == 440
    assert len(selector_calls) == 1
    assert resolver_calls == [440]
    assert len(child_runs) == 1
    child_id, child_dive_id, flat = child_runs[0]
    assert child_id == "preprocess-dive-images-440"
    assert child_dive_id == 440
    assert flat == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_returns_none_when_selector_finds_no_dive():
    activities, _, resolver_calls = _make_stubs(None, None)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-parent-empty",
            workflows=[PreprocessDiveImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessDiveImagesParentWorkflow.run,
                id=f"test-stage2-parent-empty-{uuid.uuid4()}",
                task_queue="test-stage2-parent-empty",
            )

    assert result is None
    assert not resolver_calls
    assert not child_runs


@pytest.mark.asyncio
async def test_skips_child_dispatch_when_no_clusters():
    inputs = PreprocessDiveImagesInput(
        dive_id=440,
        clusters=[],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, _, _ = _make_stubs(440, inputs)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage2-parent-empty-clusters",
            workflows=[PreprocessDiveImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessDiveImagesParentWorkflow.run,
                id=f"test-stage2-parent-empty-clusters-{uuid.uuid4()}",
                task_queue="test-stage2-parent-empty-clusters",
            )

    assert result == 440
    assert not child_runs

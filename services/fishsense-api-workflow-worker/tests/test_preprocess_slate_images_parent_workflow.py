# pylint: disable=unused-argument
"""Workflow contract test for PreprocessSlateImagesParentWorkflow."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.preprocess_slate_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DATA_PROCESSING_TASK_QUEUE,
    PreprocessSlateImagesParentWorkflow,
)
from fishsense_shared import PreprocessSlateImagesInput


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@workflow.defn(name="PreprocessSlateImagesWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PreprocessSlateImagesInput) -> None:
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(
                workflow.info().workflow_id,
                payload.dive_id,
                (list(payload.image_checksums), payload.slate_id),
            ),
            schedule_to_close_timeout=timedelta(seconds=5),
        )


def _make_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(
        workflow_id: str, dive_id: int, summary: tuple
    ) -> None:
        captures.append((workflow_id, dive_id, summary))

    return record_child_dispatch


def _make_stubs(
    selector_result: Optional[int],
    resolver_result: Optional[PreprocessSlateImagesInput],
):
    @activity.defn(name="select_next_high_priority_dive_for_slate_preprocessing_activity")
    async def stub_select() -> Optional[int]:
        return selector_result

    @activity.defn(name="resolve_slate_preprocess_inputs_activity")
    async def stub_resolve(dive_id: int) -> PreprocessSlateImagesInput:
        assert resolver_result is not None
        return resolver_result

    return [stub_select, stub_resolve]


@pytest.mark.asyncio
async def test_dispatches_child_with_deterministic_id():
    inputs = PreprocessSlateImagesInput(
        dive_id=440,
        image_checksums=["a"],
        slate_id=7,
        slate_dpi=300,
        reference_points=[(0.0, 0.0), (1.0, 1.0)],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities = _make_stubs(440, inputs)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage9-parent",
            workflows=[PreprocessSlateImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessSlateImagesParentWorkflow.run,
                id=f"test-stage9-parent-{uuid.uuid4()}",
                task_queue="test-stage9-parent",
            )

    assert result == 440
    assert len(child_runs) == 1
    child_id, child_dive_id, (checksums, slate_id) = child_runs[0]
    assert child_id == "preprocess-slate-440"
    assert child_dive_id == 440
    assert checksums == ["a"]
    assert slate_id == 7


@pytest.mark.asyncio
async def test_returns_none_when_no_dive():
    activities = _make_stubs(None, None)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage9-parent-none",
            workflows=[PreprocessSlateImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessSlateImagesParentWorkflow.run,
                id=f"test-stage9-parent-none-{uuid.uuid4()}",
                task_queue="test-stage9-parent-none",
            )

    assert result is None
    assert not child_runs


@pytest.mark.asyncio
async def test_skips_child_when_no_image_checksums():
    inputs = PreprocessSlateImagesInput(
        dive_id=440,
        image_checksums=[],
        slate_id=7,
        slate_dpi=300,
        reference_points=[(0.0, 0.0)],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities = _make_stubs(440, inputs)
    child_runs: List[tuple] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage9-parent-empty",
            workflows=[PreprocessSlateImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_make_recording_activity(child_runs)],
        ):
            result = await env.client.execute_workflow(
                PreprocessSlateImagesParentWorkflow.run,
                id=f"test-stage9-parent-empty-{uuid.uuid4()}",
                task_queue="test-stage9-parent-empty",
            )

    assert result == 440
    assert not child_runs

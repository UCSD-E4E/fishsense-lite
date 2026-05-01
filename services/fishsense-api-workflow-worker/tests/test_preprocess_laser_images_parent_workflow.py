# pylint: disable=unused-argument
"""Workflow contract test for PreprocessLaserImagesParentWorkflow.

Pins down:
  1. Selector returns None → parent returns None, no resolver/child.
  2. Selector + resolver dispatch a child workflow on the data-worker
     task queue with a deterministic id (`preprocess-laser-{dive_id}`).
  3. Resolver returning 0 checksums skips the child dispatch.

The child-workflow stub records its dispatch via an activity (rather
than module-level state) so the recording survives the workflow
sandbox boundary — module mutations from inside a workflow do not
propagate back to the test process.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows.preprocess_laser_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DATA_PROCESSING_TASK_QUEUE,
    PreprocessLaserImagesParentWorkflow,
)
from fishsense_shared import PreprocessLaserImagesInput


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]
_BBOX = [1800, 700, 2400, 1600]


@workflow.defn(name="PreprocessLaserImagesWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    """Stand-in for the data-worker's PreprocessLaserImagesWorkflow.

    Forwards its workflow_id + payload through `record_child_dispatch`
    so the contract test can observe what the parent dispatched.
    """

    @workflow.run
    async def run(self, payload: PreprocessLaserImagesInput) -> None:
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(
                workflow.info().workflow_id,
                payload.dive_id,
                list(payload.image_checksums),
            ),
            schedule_to_close_timeout=timedelta(seconds=5),
        )


def _make_recording_activity(captures: List[tuple]):
    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(
        workflow_id: str, dive_id: int, checksums: List[str]
    ) -> None:
        captures.append((workflow_id, dive_id, checksums))

    return record_child_dispatch


def _make_stub_activities(
    selector_result: Optional[int],
    resolver_result: Optional[PreprocessLaserImagesInput],
):
    selector_calls: List[None] = []
    resolver_calls: List[int] = []
    stage_calls: List[int] = []

    @activity.defn(name="select_next_high_priority_dive_for_laser_preprocessing_activity")
    async def stub_select() -> Optional[int]:
        selector_calls.append(None)
        return selector_result

    @activity.defn(name="resolve_laser_preprocess_inputs_activity")
    async def stub_resolve(dive_id: int) -> PreprocessLaserImagesInput:
        resolver_calls.append(dive_id)
        assert resolver_result is not None
        return resolver_result

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stub_stage(dive_id: int) -> None:
        stage_calls.append(dive_id)

    @activity.defn(name="archive_processed_jpegs_to_nas_activity")
    async def stub_archive(
        dive_id: int, exchange_folder: str, nas_workflow: str
    ) -> None:
        return None

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def stub_cleanup(dive_id: int) -> None:
        return None

    return (
        [stub_select, stub_resolve, stub_stage, stub_archive, stub_cleanup],
        selector_calls,
        resolver_calls,
        stage_calls,
    )


@pytest.mark.asyncio
async def test_dispatches_child_with_deterministic_id_and_correct_payload():
    inputs = PreprocessLaserImagesInput(
        dive_id=440,
        image_checksums=["a", "b", "c"],
        camera_matrix=_K,
        distortion_coefficients=_D,
        bbox=_BBOX,
    )
    activities, selector_calls, resolver_calls, stage_calls = (
        _make_stub_activities(selector_result=440, resolver_result=inputs)
    )
    child_runs: List[tuple] = []
    record = _make_recording_activity(child_runs)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-parent",
            workflows=[PreprocessLaserImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[record],
        ):
            result = await env.client.execute_workflow(
                PreprocessLaserImagesParentWorkflow.run,
                id=f"test-stage01-parent-{uuid.uuid4()}",
                task_queue="test-stage01-parent",
            )

    assert result == 440
    assert len(selector_calls) == 1
    assert resolver_calls == [440]
    assert stage_calls == [440]
    assert len(child_runs) == 1
    child_id, child_dive_id, child_checksums = child_runs[0]
    assert child_id == "preprocess-laser-440"
    assert child_dive_id == 440
    assert child_checksums == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_returns_none_when_selector_finds_no_dive():
    activities, selector_calls, resolver_calls, stage_calls = (
        _make_stub_activities(selector_result=None, resolver_result=None)
    )
    child_runs: List[tuple] = []
    record = _make_recording_activity(child_runs)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-parent-empty",
            workflows=[PreprocessLaserImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[record],
        ):
            result = await env.client.execute_workflow(
                PreprocessLaserImagesParentWorkflow.run,
                id=f"test-stage01-parent-empty-{uuid.uuid4()}",
                task_queue="test-stage01-parent-empty",
            )

    assert result is None
    assert len(selector_calls) == 1
    assert not resolver_calls
    assert not stage_calls
    assert not child_runs


@pytest.mark.asyncio
async def test_skips_child_dispatch_when_no_incomplete_images():
    inputs = PreprocessLaserImagesInput(
        dive_id=440,
        image_checksums=[],
        camera_matrix=_K,
        distortion_coefficients=_D,
        bbox=_BBOX,
    )
    activities, _, resolver_calls, stage_calls = _make_stub_activities(
        selector_result=440, resolver_result=inputs
    )
    child_runs: List[tuple] = []
    record = _make_recording_activity(child_runs)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-parent-empty-images",
            workflows=[PreprocessLaserImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[record],
        ):
            result = await env.client.execute_workflow(
                PreprocessLaserImagesParentWorkflow.run,
                id=f"test-stage01-parent-empty-images-{uuid.uuid4()}",
                task_queue="test-stage01-parent-empty-images",
            )

    assert result == 440
    assert resolver_calls == [440]
    assert not stage_calls
    assert not child_runs

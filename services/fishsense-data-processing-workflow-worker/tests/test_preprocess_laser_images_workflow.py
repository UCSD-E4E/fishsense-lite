"""Workflow contract test for PreprocessLaserImagesWorkflow.

Pins down the no-input shape: the workflow first calls the selector,
then (when a dive is returned) the resolver, then fans out one
preprocess_laser_image activity per checksum with the resolved
intrinsics + the workflow-level default bbox.
"""

import uuid
from typing import List, Optional

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.resolve_laser_preprocess_inputs_activity import (  # noqa: E501  pylint: disable=line-too-long
    LaserPreprocessInputs,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DEFAULT_LASER_BBOX,
    OUTPUT_FOLDER,
    PreprocessLaserImageInput,
    PreprocessLaserImagesWorkflow,
)


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


def _make_stubs(
    selector_result: Optional[int],
    resolver_result: Optional[LaserPreprocessInputs],
):
    selector_calls: List[None] = []
    resolver_calls: List[int] = []
    image_calls: List[PreprocessLaserImageInput] = []

    @activity.defn(name="select_next_high_priority_dive_for_laser_preprocessing_activity")
    async def stub_select() -> Optional[int]:
        selector_calls.append(None)
        return selector_result

    @activity.defn(name="resolve_laser_preprocess_inputs_activity")
    async def stub_resolve(dive_id: int) -> LaserPreprocessInputs:
        resolver_calls.append(dive_id)
        assert resolver_result is not None
        return resolver_result

    @activity.defn(name="preprocess_laser_image")
    async def stub_preprocess(payload: PreprocessLaserImageInput) -> None:
        image_calls.append(payload)

    return (
        [stub_select, stub_resolve, stub_preprocess],
        selector_calls,
        resolver_calls,
        image_calls,
    )


@pytest.mark.asyncio
async def test_workflow_drains_one_dive_and_fans_out_per_image():
    inputs = LaserPreprocessInputs(
        dive_id=440,
        image_checksums=["a", "b", "c"],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, selector_calls, resolver_calls, image_calls = _make_stubs(
        selector_result=440, resolver_result=inputs
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01",
            workflows=[PreprocessLaserImagesWorkflow],
            activities=activities,
        ):
            result = await env.client.execute_workflow(
                PreprocessLaserImagesWorkflow.run,
                id=f"test-stage01-{uuid.uuid4()}",
                task_queue="test-stage01",
            )

    assert result == 440
    assert len(selector_calls) == 1
    assert resolver_calls == [440]
    assert len(image_calls) == 3
    assert {c.checksum for c in image_calls} == {"a", "b", "c"}
    for call in image_calls:
        assert tuple(call.bbox) == DEFAULT_LASER_BBOX
        assert call.output_folder == OUTPUT_FOLDER
        assert call.camera_matrix == _K
        assert call.distortion_coefficients == _D


@pytest.mark.asyncio
async def test_workflow_returns_none_when_selector_finds_no_dive():
    activities, selector_calls, resolver_calls, image_calls = _make_stubs(
        selector_result=None, resolver_result=None
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-empty-queue",
            workflows=[PreprocessLaserImagesWorkflow],
            activities=activities,
        ):
            result = await env.client.execute_workflow(
                PreprocessLaserImagesWorkflow.run,
                id=f"test-stage01-empty-queue-{uuid.uuid4()}",
                task_queue="test-stage01-empty-queue",
            )

    assert result is None
    assert len(selector_calls) == 1
    assert not resolver_calls
    assert not image_calls


@pytest.mark.asyncio
async def test_workflow_with_no_incomplete_images_skips_per_image_fanout():
    inputs = LaserPreprocessInputs(
        dive_id=440,
        image_checksums=[],
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    activities, _, resolver_calls, image_calls = _make_stubs(
        selector_result=440, resolver_result=inputs
    )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-empty-images",
            workflows=[PreprocessLaserImagesWorkflow],
            activities=activities,
        ):
            result = await env.client.execute_workflow(
                PreprocessLaserImagesWorkflow.run,
                id=f"test-stage01-empty-images-{uuid.uuid4()}",
                task_queue="test-stage01-empty-images",
            )

    assert result == 440
    assert resolver_calls == [440]
    assert not image_calls

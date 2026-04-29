"""Workflow contract test for PreprocessSlateImagesWorkflow."""

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.preprocess_slate_images_workflow import (
    PreprocessSlateImageInput,
    PreprocessSlateImagesInput,
    PreprocessSlateImagesWorkflow,
)


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]
_REF_POINTS = [(100.0, 200.0), (300.0, 400.0)]


@pytest.mark.asyncio
async def test_workflow_fans_out_one_activity_per_image_with_correct_args():
    calls: List[PreprocessSlateImageInput] = []

    @activity.defn(name="preprocess_slate_image")
    async def stub(payload: PreprocessSlateImageInput) -> None:
        calls.append(payload)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage9",
            workflows=[PreprocessSlateImagesWorkflow],
            activities=[stub],
        ):
            await env.client.execute_workflow(
                PreprocessSlateImagesWorkflow.run,
                PreprocessSlateImagesInput(
                    dive_id=383,
                    image_checksums=["a", "b"],
                    slate_id=10,
                    slate_dpi=300,
                    reference_points=_REF_POINTS,
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage9-{uuid.uuid4()}",
                task_queue="test-stage9",
            )

    assert {c.checksum for c in calls} == {"a", "b"}
    for c in calls:
        assert c.output_folder == "preprocess_slate_images_jpeg"
        assert c.slate_id == 10
        assert c.slate_dpi == 300
        assert [tuple(p) for p in c.reference_points] == _REF_POINTS
        assert c.camera_matrix == _K
        assert c.distortion_coefficients == _D


@pytest.mark.asyncio
async def test_workflow_with_no_images_makes_no_activity_calls():
    calls: List[PreprocessSlateImageInput] = []

    @activity.defn(name="preprocess_slate_image")
    async def stub(payload: PreprocessSlateImageInput) -> None:
        calls.append(payload)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage9-empty",
            workflows=[PreprocessSlateImagesWorkflow],
            activities=[stub],
        ):
            await env.client.execute_workflow(
                PreprocessSlateImagesWorkflow.run,
                PreprocessSlateImagesInput(
                    dive_id=383,
                    image_checksums=[],
                    slate_id=10,
                    slate_dpi=300,
                    reference_points=_REF_POINTS,
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage9-empty-{uuid.uuid4()}",
                task_queue="test-stage9-empty",
            )

    assert not calls

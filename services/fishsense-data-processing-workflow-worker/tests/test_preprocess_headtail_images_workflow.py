"""Workflow contract test for PreprocessHeadtailImagesWorkflow."""

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.preprocess_headtail_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessHeadtailImageInput,
    PreprocessHeadtailImagesInput,
    PreprocessHeadtailImagesWorkflow,
)


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_workflow_fans_out_one_activity_per_image_with_correct_args():
    calls: List[PreprocessHeadtailImageInput] = []

    @activity.defn(name="preprocess_headtail_image")
    async def stub(payload: PreprocessHeadtailImageInput) -> None:
        calls.append(payload)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage51",
            workflows=[PreprocessHeadtailImagesWorkflow],
            activities=[stub],
        ):
            await env.client.execute_workflow(
                PreprocessHeadtailImagesWorkflow.run,
                PreprocessHeadtailImagesInput(
                    dive_id=383,
                    image_checksums=["a", "b", "c"],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage51-{uuid.uuid4()}",
                task_queue="test-stage51",
            )

    assert {c.checksum for c in calls} == {"a", "b", "c"}
    for c in calls:
        assert c.output_folder == "preprocess_headtail_jpeg"
        assert c.camera_matrix == _K
        assert c.distortion_coefficients == _D


@pytest.mark.asyncio
async def test_workflow_with_no_images_makes_no_activity_calls():
    calls: List[PreprocessHeadtailImageInput] = []

    @activity.defn(name="preprocess_headtail_image")
    async def stub(payload: PreprocessHeadtailImageInput) -> None:
        calls.append(payload)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage51-empty",
            workflows=[PreprocessHeadtailImagesWorkflow],
            activities=[stub],
        ):
            await env.client.execute_workflow(
                PreprocessHeadtailImagesWorkflow.run,
                PreprocessHeadtailImagesInput(
                    dive_id=383,
                    image_checksums=[],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage51-empty-{uuid.uuid4()}",
                task_queue="test-stage51-empty",
            )

    assert not calls

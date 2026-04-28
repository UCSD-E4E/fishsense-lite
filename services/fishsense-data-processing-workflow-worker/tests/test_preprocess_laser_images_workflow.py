"""Workflow contract test for PreprocessLaserImagesWorkflow.

Runs the workflow against an in-process Temporal test server and stubs
the activity to record its inputs. We don't assert on real image work
here — only on the workflow's fanout shape and per-image arg propagation.
"""

import asyncio
import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (
    PreprocessLaserImageInput,
    PreprocessLaserImagesInput,
    PreprocessLaserImagesWorkflow,
)


_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]
_BBOX = (1800, 700, 2400, 1600)


@pytest.mark.asyncio
async def test_workflow_fans_out_one_activity_per_image_with_correct_args():
    calls: List[PreprocessLaserImageInput] = []

    @activity.defn(name="preprocess_laser_image")
    async def stub_preprocess_laser_image(
        input: PreprocessLaserImageInput,
    ) -> None:
        calls.append(input)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01",
            workflows=[PreprocessLaserImagesWorkflow],
            activities=[stub_preprocess_laser_image],
        ):
            await env.client.execute_workflow(
                PreprocessLaserImagesWorkflow.run,
                PreprocessLaserImagesInput(
                    dive_id=440,
                    image_checksums=["a", "b", "c"],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                    bbox=list(_BBOX),
                ),
                id=f"test-stage01-{uuid.uuid4()}",
                task_queue="test-stage01",
            )

    assert len(calls) == 3
    by_checksum = {c.checksum: c for c in calls}
    assert set(by_checksum) == {"a", "b", "c"}

    for c in calls:
        assert tuple(c.bbox) == _BBOX
        assert c.camera_matrix == _K
        assert c.distortion_coefficients == _D
        assert c.output_folder == "preprocess_jpeg"


@pytest.mark.asyncio
async def test_workflow_with_no_images_makes_no_activity_calls():
    calls: List[PreprocessLaserImageInput] = []

    @activity.defn(name="preprocess_laser_image")
    async def stub_preprocess_laser_image(
        input: PreprocessLaserImageInput,
    ) -> None:
        calls.append(input)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-empty",
            workflows=[PreprocessLaserImagesWorkflow],
            activities=[stub_preprocess_laser_image],
        ):
            await env.client.execute_workflow(
                PreprocessLaserImagesWorkflow.run,
                PreprocessLaserImagesInput(
                    dive_id=440,
                    image_checksums=[],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                    bbox=list(_BBOX),
                ),
                id=f"test-stage01-empty-{uuid.uuid4()}",
                task_queue="test-stage01-empty",
            )

    assert calls == []

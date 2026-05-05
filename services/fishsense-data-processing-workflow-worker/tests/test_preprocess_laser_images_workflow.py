"""Workflow contract test for PreprocessLaserImagesWorkflow.

Runs the workflow against an in-process Temporal test server and stubs
the activity to record its payloads. We don't assert on real image work
here — only on the workflow's fanout shape and per-image arg propagation.
"""

import uuid
from datetime import timedelta
from typing import List, Optional, Tuple

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
        payload: PreprocessLaserImageInput,
    ) -> None:
        calls.append(payload)

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
async def test_workflow_uses_start_to_close_not_schedule_to_close():
    """See PreprocessHeadtailImagesWorkflow's matching test — same fan-out
    shape, same prod failure mode (queue wait > schedule_to_close on dives
    with image counts >> data-worker pool size)."""
    timeouts: List[Tuple[Optional[timedelta], Optional[timedelta]]] = []

    @activity.defn(name="preprocess_laser_image")
    async def stub_preprocess_laser_image(payload: PreprocessLaserImageInput) -> None:  # pylint: disable=unused-argument
        info = activity.info()
        timeouts.append((info.start_to_close_timeout, info.schedule_to_close_timeout))

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage01-timeouts",
            workflows=[PreprocessLaserImagesWorkflow],
            activities=[stub_preprocess_laser_image],
        ):
            await env.client.execute_workflow(
                PreprocessLaserImagesWorkflow.run,
                PreprocessLaserImagesInput(
                    dive_id=440,
                    image_checksums=["a"],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                    bbox=list(_BBOX),
                ),
                id=f"test-stage01-timeouts-{uuid.uuid4()}",
                task_queue="test-stage01-timeouts",
            )

    assert len(timeouts) == 1
    start_to_close, schedule_to_close = timeouts[0]
    assert start_to_close is not None and start_to_close > timedelta(0)
    if schedule_to_close is not None and schedule_to_close > timedelta(0):
        assert schedule_to_close > start_to_close


@pytest.mark.asyncio
async def test_workflow_with_no_images_makes_no_activity_calls():
    calls: List[PreprocessLaserImageInput] = []

    @activity.defn(name="preprocess_laser_image")
    async def stub_preprocess_laser_image(
        payload: PreprocessLaserImageInput,
    ) -> None:
        calls.append(payload)

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

    assert not calls

"""Workflow contract test for PreprocessHeadtailImagesWorkflow."""

import uuid
from datetime import timedelta
from typing import List, Optional, Tuple

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
async def test_workflow_uses_start_to_close_not_schedule_to_close():
    """Per-image preprocess activities must be timed by execution duration,
    not queue+execution. With fan-out across many images on a small worker
    pool, schedule_to_close_timeout would tick down while activities sit in
    the task queue waiting for an executor slot — the dive-76 head/tail run
    failed exactly this way in prod (activity scheduled at event 16, started
    at event 352, hit the 5-minute schedule_to_close before completing).
    """
    timeouts: List[Tuple[Optional[timedelta], Optional[timedelta]]] = []

    @activity.defn(name="preprocess_headtail_image")
    async def stub(payload: PreprocessHeadtailImageInput) -> None:  # pylint: disable=unused-argument
        info = activity.info()
        timeouts.append((info.start_to_close_timeout, info.schedule_to_close_timeout))

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-stage51-timeouts",
            workflows=[PreprocessHeadtailImagesWorkflow],
            activities=[stub],
        ):
            await env.client.execute_workflow(
                PreprocessHeadtailImagesWorkflow.run,
                PreprocessHeadtailImagesInput(
                    dive_id=383,
                    image_checksums=["a"],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-stage51-timeouts-{uuid.uuid4()}",
                task_queue="test-stage51-timeouts",
            )

    assert len(timeouts) == 1
    start_to_close, schedule_to_close = timeouts[0]
    assert start_to_close is not None and start_to_close > timedelta(0), (
        "per-image activity must set start_to_close_timeout — execution "
        "budget per image, not queue+execution"
    )
    if schedule_to_close is not None and schedule_to_close > timedelta(0):
        assert schedule_to_close > start_to_close, (
            "schedule_to_close (queue+execution) must not be the binding "
            "constraint on per-image execution; it should be looser than "
            "start_to_close, since fan-out wait time is unbounded"
        )


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

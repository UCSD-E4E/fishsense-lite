"""Role split: which queue a worker polls, and what it registers.

The data-worker used to be one process on one queue whose pod requested
``nvidia.com/gpu: 1``, so every stage — nine CPU activities included — was
gated on NRP scheduling a Turing-or-newer GPU. It is now split by role: the
``cpu`` role serves `fishsense_data_processing_queue` and the ``gpu`` role
serves `fishsense_data_processing_gpu_queue`, which only the two torch
inference stages land on.

The load-bearing test here is `test_every_registration_lands_in_exactly_one
_role`. The split is two hand-maintained lists, and the failure mode of
getting it wrong is silent in both directions: an activity in neither list is
never registered anywhere, so its workflow's `execute_activity` sits pending
until the schedule-to-close timeout with nothing logged; an activity in both
is registered on a queue whose pod may have no GPU. Neither shows up until a
dive is being processed in prod.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
from fishsense_shared import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)
from temporalio.testing import WorkflowEnvironment

from fishsense_data_processing_workflow_worker import roles
from fishsense_data_processing_workflow_worker.activities.predict_slate_image import (
    predict_slate_image,
)
from fishsense_data_processing_workflow_worker.workflows.predict_slate_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictSlateImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.worker import (
    build_worker,
    build_workers,
)


def test_role_queues_are_distinct_and_come_from_the_shared_contract():
    assert roles.queue_for_role(roles.ROLE_CPU) == DATA_PROCESSING_TASK_QUEUE
    assert roles.queue_for_role(roles.ROLE_GPU) == DATA_PROCESSING_GPU_TASK_QUEUE
    assert DATA_PROCESSING_TASK_QUEUE != DATA_PROCESSING_GPU_TASK_QUEUE


def test_every_registration_lands_in_exactly_one_role():
    """No activity or workflow may be in both role lists or in neither.

    Both mistakes are silent in prod — see the module docstring.
    """
    assert not set(roles.CPU_ACTIVITIES) & set(roles.GPU_ACTIVITIES)
    assert not set(roles.CPU_WORKFLOWS) & set(roles.GPU_WORKFLOWS)
    assert set(roles.CPU_ACTIVITIES) | set(roles.GPU_ACTIVITIES) == set(
        roles.ALL_ACTIVITIES
    )
    assert set(roles.CPU_WORKFLOWS) | set(roles.GPU_WORKFLOWS) == set(
        roles.ALL_WORKFLOWS
    )


def test_gpu_role_holds_exactly_the_torch_inference_stages():
    """The GPU queue carries the torch models and nothing else. Anything else
    there would wait on the GPU-capacity handshake for no benefit.

    `predict_headtail_image` joined them when the head/tail stage moved from
    the fishsense-core Mask R-CNN to SAM3 — a reversal from its original CPU
    design, and the only cost of that backend choice. It is cheap per image
    (~0.6 s against the 2.9 s CPU design it replaced), so it holds a card
    briefly rather than continuously."""
    assert {activity.__name__ for activity in roles.GPU_ACTIVITIES} == {
        "predict_headtail_image",
        "predict_laser_image",
        "predict_slate_image",
    }
    assert {workflow.__name__ for workflow in roles.GPU_WORKFLOWS} == {
        "PredictHeadtailImagesWorkflow",
        "PredictLaserImagesWorkflow",
        "PredictSlateImagesWorkflow",
    }


def test_slate_prediction_prefers_the_gpu_without_requiring_one():
    """`predict_slate_image` is on the GPU queue even though it does not need a
    GPU (fishsense-core measures the mask at 202 ms/frame on CPU).

    That is safe precisely because the GPU queue means *prefer*, not *require*:
    it is served by the GPU Deployment or, when that can't start, a CPU-only
    one. So the stage gets the card the laser detector already needs, and is
    never gated on there being one.
    """
    assert predict_slate_image in roles.GPU_ACTIVITIES
    assert PredictSlateImagesWorkflow in roles.GPU_WORKFLOWS


@pytest.mark.parametrize(
    ("role", "expected_queue"),
    [
        (roles.ROLE_CPU, DATA_PROCESSING_TASK_QUEUE),
        (roles.ROLE_GPU, DATA_PROCESSING_GPU_TASK_QUEUE),
    ],
)
@pytest.mark.asyncio
async def test_build_worker_polls_the_queue_for_its_role(role, expected_queue):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = build_worker(env.client, executor, role=role)
            assert worker.config()["task_queue"] == expected_queue


@pytest.mark.asyncio
async def test_build_worker_rejects_the_all_role():
    """``all`` is two queues, so it cannot produce one Worker. Callers that
    want it must use ``build_workers``; silently returning just the CPU half
    would leave the GPU queue unserved."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(ValueError):
                build_worker(env.client, executor, role=roles.ROLE_ALL)


@pytest.mark.asyncio
async def test_build_worker_rejects_an_unknown_role():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(ValueError):
                build_worker(env.client, executor, role="tpu")


@pytest.mark.asyncio
async def test_build_workers_all_serves_both_queues_in_one_process():
    """The devcontainer default. One process, both queues — so local runs and
    the integration tests behave exactly as they did before the split."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            workers = build_workers(env.client, executor, role=roles.ROLE_ALL)
            assert {worker.config()["task_queue"] for worker in workers} == {
                DATA_PROCESSING_TASK_QUEUE,
                DATA_PROCESSING_GPU_TASK_QUEUE,
            }


@pytest.mark.asyncio
async def test_build_workers_single_role_returns_one_worker():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            workers = build_workers(env.client, executor, role=roles.ROLE_GPU)
            assert len(workers) == 1
            assert (
                workers[0].config()["task_queue"] == DATA_PROCESSING_GPU_TASK_QUEUE
            )

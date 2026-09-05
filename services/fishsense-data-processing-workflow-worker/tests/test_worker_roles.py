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

import itertools
from concurrent.futures import ThreadPoolExecutor

import pytest
from fishsense_shared import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_LIGHT_TASK_QUEUE,
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
    assert roles.queue_for_role(roles.ROLE_LIGHT) == DATA_PROCESSING_LIGHT_TASK_QUEUE
    assert (
        len(
            {
                DATA_PROCESSING_TASK_QUEUE,
                DATA_PROCESSING_GPU_TASK_QUEUE,
                DATA_PROCESSING_LIGHT_TASK_QUEUE,
            }
        )
        == 3
    )


def test_every_registration_lands_in_exactly_one_role():
    """No activity or workflow may be in two role lists or in none.

    Both mistakes are silent in prod — see the module docstring. Written over
    every pair rather than the one CPU/GPU pair it started as, so adding a
    fourth role cannot quietly escape the check.
    """
    activity_lists = {
        roles.ROLE_CPU: set(roles.CPU_ACTIVITIES),
        roles.ROLE_GPU: set(roles.GPU_ACTIVITIES),
        roles.ROLE_LIGHT: set(roles.LIGHT_ACTIVITIES),
    }
    workflow_lists = {
        roles.ROLE_CPU: set(roles.CPU_WORKFLOWS),
        roles.ROLE_GPU: set(roles.GPU_WORKFLOWS),
        roles.ROLE_LIGHT: set(roles.LIGHT_WORKFLOWS),
    }

    for lists in (activity_lists, workflow_lists):
        for left, right in itertools.combinations(sorted(lists), 2):
            overlap = lists[left] & lists[right]
            assert not overlap, f"registered on both {left} and {right}: {overlap}"

    assert set().union(*activity_lists.values()) == set(roles.ALL_ACTIVITIES)
    assert set().union(*workflow_lists.values()) == set(roles.ALL_WORKFLOWS)


def test_light_role_holds_the_stages_with_no_per_image_fan_out():
    """The light queue exists so a sub-second fit never queues behind a
    multi-GB rawpy decode.

    The CPU worker runs `max_concurrent_activities = 2` — a cap that is the
    scar of a real OOM crash-loop, since each per-image activity peaks at
    1-3 GB decoding a full-res `.ORF`. Two slots means one 34-image preprocess
    dispatch owns the whole queue, and on 2026-09-04 that expired three of the
    auto-accept drain's first four firings and both of laser calibration's.

    These six stages are the ones that can share a pod safely: they fetch rows
    from fishsense-api, do numpy, and write rows back. No image bytes, no
    object-store traffic, no fan-out — so their memory is flat and their
    concurrency cap has nothing to do with the CPU worker's.
    """
    assert {activity.__name__ for activity in roles.LIGHT_ACTIVITIES} == {
        "cluster_dive_frames",
        "compute_laser_depths_activity",
        "evaluate_laser_auto_accept_activity",
        "measure_fish_activity",
        "perform_laser_calibration_activity",
        "validate_laser_labels_for_dive_activity",
    }
    assert {workflow.__name__ for workflow in roles.LIGHT_WORKFLOWS} == {
        "ComputeLaserDepthsWorkflow",
        "DiveFrameClusteringWorkflow",
        "EvaluateLaserAutoAcceptWorkflow",
        "MeasureFishWorkflow",
        "PerformLaserCalibrationWorkflow",
        "ValidateLaserLabelsForDiveWorkflow",
    }


def test_cpu_role_keeps_exactly_the_per_image_fan_out_stages():
    """The converse, so moving a stage to the light queue cannot silently take
    a memory-heavy one with it. These four are the rawpy decoders."""
    assert {activity.__name__ for activity in roles.CPU_ACTIVITIES} == {
        "preprocess_headtail_image",
        "preprocess_laser_image",
        "preprocess_slate_image",
        "preprocess_species_image",
    }


def test_gpu_role_holds_exactly_the_torch_inference_stages():
    """The GPU queue carries the two torch models and nothing else. Anything
    else there would wait on the GPU-capacity handshake for no benefit."""
    assert {activity.__name__ for activity in roles.GPU_ACTIVITIES} == {
        "predict_laser_image",
        "predict_slate_image",
    }
    assert {workflow.__name__ for workflow in roles.GPU_WORKFLOWS} == {
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
        (roles.ROLE_LIGHT, DATA_PROCESSING_LIGHT_TASK_QUEUE),
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
    """``all`` is every queue, so it cannot produce one Worker. Callers
    that want it must use ``build_workers``; silently returning just the CPU
    half would leave the others unserved."""
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
async def test_build_workers_all_serves_every_queue_in_one_process():
    """The devcontainer default. One process, every queue — so local runs and
    the integration tests behave exactly as they did before the split. This is
    the test that catches a new role being added without teaching ``all``
    about it, which would leave its queue unserved locally."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        with ThreadPoolExecutor(max_workers=1) as executor:
            workers = build_workers(env.client, executor, role=roles.ROLE_ALL)
            assert {worker.config()["task_queue"] for worker in workers} == {
                DATA_PROCESSING_TASK_QUEUE,
                DATA_PROCESSING_GPU_TASK_QUEUE,
                DATA_PROCESSING_LIGHT_TASK_QUEUE,
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

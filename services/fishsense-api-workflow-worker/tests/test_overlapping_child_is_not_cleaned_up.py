# pylint: disable=unused-argument,protected-access
"""An overlapping child must not have its inputs deleted underneath it.

`dispatch_child` uses ALLOW_DUPLICATE, so `WorkflowAlreadyStartedError` is
reachable only while a prior child with the same id is still *running* -- a
manual run overlapping the hourly schedule. The catch was deliberate and the
docstring said the caller "continues to its cleanup rather than failing the
firing". That is the bug: cleanup deletes the dive's staged raw `.ORF`s from
Garage, and the running child is still reading them.

Observed in prod 2026-09-07 on dive 442. A manual stage-0.1 parent started a
child at 06:42 which scheduled all 259 per-image activities. The 07:00
scheduled firing selected the same dive, was refused the duplicate dispatch,
and continued: it deleted 984 raw objects and cleared 515 reprocess flags. The
child died at 07:40 with

    An error occurred (NoSuchKey) when calling the GetObject operation

having redrawn 2 of 259 frames, and the flags were already down so nothing
brought the rest back.

So a firing that did not dispatch must do nothing else: the run that owns the
child will clean up after it, and will clear the flags for the frames it
actually redrew.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import List, Optional

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.reprocess_scope import (
    ClearReprocessFlagsInput,
)
from fishsense_api_workflow_worker.workflows.preprocess_laser_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessLaserImagesParentWorkflow,
)
from fishsense_shared import PreprocessLaserImagesInput

_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]
_DIVE = 442


@workflow.defn(name="PreprocessLaserImagesWorkflow")
class _SlowChild:
    # pylint: disable=too-few-public-methods
    """Stands in for a child that is still rendering when the next firing lands."""

    @workflow.run
    async def run(self, payload: PreprocessLaserImagesInput) -> None:
        await asyncio.sleep(3600)


def _stubs(cleanup_calls: List[int], clear_calls: List[int], stage_calls: List[int]):
    inputs = PreprocessLaserImagesInput(
        dive_id=_DIVE,
        image_checksums=["a", "b"],
        camera_matrix=_K,
        distortion_coefficients=_D,
        bbox=[0, 0, 10, 10],
    )

    @activity.defn(name="select_next_high_priority_dive_for_laser_preprocessing_activity")
    async def stub_select() -> Optional[int]:
        return _DIVE

    @activity.defn(name="resolve_laser_preprocess_inputs_activity")
    async def stub_resolve(dive_id: int) -> PreprocessLaserImagesInput:
        return inputs

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stub_stage(dive_id: int) -> None:
        stage_calls.append(dive_id)

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def stub_cleanup(dive_id: int) -> None:
        cleanup_calls.append(dive_id)

    @activity.defn(name="ensure_data_worker_running_activity")
    async def stub_ensure() -> int:
        return 0

    @activity.defn(name="clear_laser_reprocess_flags_activity")
    async def stub_clear(payload: ClearReprocessFlagsInput) -> int:
        clear_calls.append(payload.dive_id)
        return 0

    return [stub_select, stub_resolve, stub_stage, stub_cleanup, stub_ensure, stub_clear]


@pytest.mark.asyncio
async def test_firing_that_could_not_dispatch_does_not_clean_up_or_clear():
    cleanup_calls: List[int] = []
    clear_calls: List[int] = []
    stage_calls: List[int] = []

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-overlap-parent",
            workflows=[PreprocessLaserImagesParentWorkflow, _SlowChild],
            activities=_stubs(cleanup_calls, clear_calls, stage_calls),
        ):
            # A child for this dive is already running -- the manual run.
            await env.client.start_workflow(
                _SlowChild.run,
                PreprocessLaserImagesInput(
                    dive_id=_DIVE, image_checksums=["a", "b"],
                    camera_matrix=_K, distortion_coefficients=_D, bbox=[0, 0, 10, 10],
                ),
                id=f"preprocess-laser-{_DIVE}",
                task_queue="test-overlap-parent",
            )

            result = await env.client.execute_workflow(
                PreprocessLaserImagesParentWorkflow.run,
                id="overlap-parent-run",
                task_queue="test-overlap-parent",
                execution_timeout=timedelta(seconds=60),
            )

    assert result == _DIVE
    assert not cleanup_calls, (
        "deleting the dive's raw scratch while another child is reading it is "
        "what killed prod dive 442's render with NoSuchKey"
    )
    assert not clear_calls, (
        "the flags belong to the run that is actually redrawing; clearing them "
        "here loses the request with nothing to show for it"
    )

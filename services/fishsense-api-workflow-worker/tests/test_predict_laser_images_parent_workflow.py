# pylint: disable=unused-argument
"""Workflow contract test for PredictLaserImagesParentWorkflow.

Pins down:
  1. Selector None -> parent returns None, no resolver/child.
  2. Full path: dispatches the child on the data-worker queue with the
     deterministic id `predict-laser-{dive_id}`, then persists the child's
     results and returns the dive_id.
  3. Resolver returning 0 images skips the child + persist.

The child stub returns predictions; the persist stub records what the parent
handed it (via an activity, so it survives the workflow sandbox boundary).
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_shared import (
    LaserPredictionResult,
    PredictLaserImage,
    PredictLaserImagesInput,
)
from fishsense_api_workflow_worker.workflows.predict_laser_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictLaserImagesParentWorkflow,
)

_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@workflow.defn(name="PredictLaserImagesWorkflow")
class _StubChild:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, payload: PredictLaserImagesInput) -> List[LaserPredictionResult]:
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(workflow.info().workflow_id, payload.dive_id),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        return [
            LaserPredictionResult(
                image_id=img.image_id, x=1.0, y=2.0, confidence=0.9
            )
            for img in payload.images
        ]


def _stubs(dive_id, images, child_ids, persisted, task_queues):
    @activity.defn(name="select_next_high_priority_dive_for_laser_prediction_activity")
    async def selector() -> int | None:
        return dive_id

    @activity.defn(name="resolve_laser_predict_inputs_activity")
    async def resolver(d: int) -> PredictLaserImagesInput:
        return PredictLaserImagesInput(
            dive_id=d,
            images=[PredictLaserImage(image_id=i, checksum=c) for i, c in images],
            camera_matrix=_K,
            distortion_coefficients=_D,
            wavelength=None,
        )

    @activity.defn(name="ensure_data_worker_running_activity")
    async def ensure() -> None:
        return None

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stage(d: int) -> None:
        return None

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def cleanup(d: int) -> None:
        return None

    @activity.defn(name="persist_laser_predictions_activity")
    async def persist(results: list) -> int:
        persisted.extend(results)
        return len(results)

    @activity.defn(name="_record_child_dispatch")
    async def record(workflow_id: str, d: int) -> None:
        child_ids.append(workflow_id)

    return [selector, resolver, ensure, stage, cleanup, persist, record]


async def _run(env, dive_id, images, child_ids, persisted, task_queues):
    async with Worker(
        env.client,
        task_queue="test-predict-parent",
        workflows=[PredictLaserImagesParentWorkflow, _StubChild],
        activities=_stubs(dive_id, images, child_ids, persisted, task_queues),
    ):
        return await env.client.execute_workflow(
            PredictLaserImagesParentWorkflow.run,
            id=f"predict-parent-{uuid.uuid4()}",
            task_queue="test-predict-parent",
        )


@pytest.mark.asyncio
async def test_selector_none_returns_none():
    child_ids: List[str] = []
    persisted: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, None, [], child_ids, persisted, [])
    assert result is None
    assert not child_ids
    assert not persisted


@pytest.mark.asyncio
async def test_full_path_dispatches_child_and_persists():
    child_ids: List[str] = []
    persisted: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, 440, [(1, "a"), (2, "b")], child_ids, persisted, []
        )
    assert result == 440
    assert child_ids == ["predict-laser-440"]  # deterministic id
    assert {p["image_id"] if isinstance(p, dict) else p.image_id for p in persisted} == {
        1,
        2,
    }


@pytest.mark.asyncio
async def test_no_images_skips_child_and_persist():
    child_ids: List[str] = []
    persisted: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, 440, [], child_ids, persisted, [])
    assert result == 440
    assert not child_ids
    assert not persisted

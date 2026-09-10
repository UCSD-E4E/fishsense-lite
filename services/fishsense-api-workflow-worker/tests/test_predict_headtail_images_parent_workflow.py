# pylint: disable=unused-argument
"""Workflow contract test for PredictHeadtailImagesParentWorkflow.

Pins the step the stage's visibility depends on: the backfill runs on every
firing that dispatched a child, not only when there was something to persist.

That distinction is the whole bug. The backfill is also what points the LS
project's `model_version` at the tier the dive holds, and without that every
attached prediction is invisible to the labeler. An all-fallback dive
re-offered on CPU capacity returns nothing but `NO_UPGRADE_AVAILABLE`, so a
gate on "something to persist" would skip precisely the dive whose project may
still be showing nothing at all.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import List

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows._dispatch import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
)
from fishsense_api_workflow_worker.workflows.predict_headtail_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictHeadtailImagesParentWorkflow,
)
from fishsense_shared.preprocess_contracts import (
    HEADTAIL_STATUS_NO_UPGRADE_AVAILABLE,
    HeadtailPredictionResult,
    PredictHeadtailImage,
    PredictHeadtailImagesInput,
)

DIVE = 94
_PERSISTED: List[List[HeadtailPredictionResult]] = []
_BACKFILLED: List[int] = []
#: What the stub child returns. Module-level because Temporal refuses a
#: workflow class defined inside a function -- but the child reads it through
#: an *activity*, never directly: a workflow body runs in Temporal's sandbox,
#: which re-imports this module and so sees a fresh, empty list. Reading it
#: inline made the child return nothing, and the all-skip test then passed
#: vacuously while pinning nothing at all.
_CHILD_RESULTS: List[HeadtailPredictionResult] = []


@pytest.fixture(autouse=True)
def _reset():
    _PERSISTED.clear()
    _BACKFILLED.clear()
    _CHILD_RESULTS.clear()


def _inputs() -> PredictHeadtailImagesInput:
    return PredictHeadtailImagesInput(
        dive_id=DIVE,
        jpeg_folder="preprocess_headtail_jpeg",
        images=[
            PredictHeadtailImage(
                image_id=1,
                checksum="abc",
                laser_points=[[10.0, 10.0]],
                laser_label_ids=[5],
            )
        ],
    )


@workflow.defn(name="PredictHeadtailImagesWorkflow")
class _StubChild:  # pylint: disable=too-few-public-methods
    """Returns whatever the test staged in `_CHILD_RESULTS`, via an activity —
    see the note on `_CHILD_RESULTS` for why it cannot simply read it."""

    @workflow.run
    async def run(self, payload) -> List[HeadtailPredictionResult]:
        return await workflow.execute_activity(
            "_staged_child_results",
            schedule_to_close_timeout=timedelta(seconds=10),
            result_type=List[HeadtailPredictionResult],
        )


def _activities():
    @activity.defn(name="select_next_high_priority_dive_for_headtail_prediction_activity")
    async def select() -> int:
        return DIVE

    @activity.defn(name="resolve_headtail_predict_inputs_activity")
    async def resolve(dive_id: int) -> PredictHeadtailImagesInput:
        return _inputs()

    @activity.defn(name="ensure_gpu_worker_running_activity")
    async def ensure_gpu() -> str:
        return "gpu"

    @activity.defn(name="persist_headtail_predictions_activity")
    async def persist(results: List[HeadtailPredictionResult]) -> int:
        _PERSISTED.append(list(results))
        return len(results)

    @activity.defn(name="backfill_headtail_predictions_for_dive_activity")
    async def backfill(dive_id: int) -> int:
        _BACKFILLED.append(dive_id)
        return 0

    @activity.defn(name="_staged_child_results")
    async def staged_child_results() -> List[HeadtailPredictionResult]:
        return list(_CHILD_RESULTS)

    return [select, resolve, ensure_gpu, persist, backfill, staged_child_results]


async def _run(child_results):
    _CHILD_RESULTS.extend(child_results)
    activities = _activities()
    async with await WorkflowEnvironment.start_time_skipping() as env:
        # Two workers, and the split is the point. The stub child is registered
        # on the data-processing **GPU** queue and nowhere else, so a
        # regression that dispatched it to the CPU queue would hang rather than
        # pass -- the same contract `test_predict_laser_images_parent_workflow`
        # pins. Registering the child alongside the parent instead is not a
        # harmless simplification: it makes the child reachable from a queue
        # the parent never uses, and the run then hangs on a child nobody
        # serves.
        async with Worker(
            env.client,
            task_queue="test-predict-headtail",
            workflows=[PredictHeadtailImagesParentWorkflow],
            activities=activities,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_GPU_TASK_QUEUE,
            workflows=[_StubChild],
            activities=activities,
        ):
            return await env.client.execute_workflow(
                PredictHeadtailImagesParentWorkflow.run,
                id=f"predict-headtail-parent-{uuid.uuid4()}",
                task_queue="test-predict-headtail",
            )


def _predicted(image_id: int) -> HeadtailPredictionResult:
    return HeadtailPredictionResult(
        image_id=image_id,
        status="predicted",
        head_x=1.0,
        head_y=2.0,
        tail_x=3.0,
        tail_y=4.0,
        predictor_version=2,
    )


def _skipped(image_id: int) -> HeadtailPredictionResult:
    return HeadtailPredictionResult(
        image_id=image_id,
        status=HEADTAIL_STATUS_NO_UPGRADE_AVAILABLE,
        predictor_version=-1,
    )


@pytest.mark.asyncio
async def test_persists_and_backfills_a_normal_run():
    assert await _run([_predicted(1)]) == DIVE
    assert [r.image_id for r in _PERSISTED[0]] == [1]
    assert _BACKFILLED == [DIVE]


@pytest.mark.asyncio
async def test_skips_are_never_persisted():
    """Persisting a skip marker would blank a perfectly good prediction."""
    assert await _run([_predicted(1), _skipped(2)]) == DIVE
    assert [r.image_id for r in _PERSISTED[0]] == [1]


@pytest.mark.asyncio
async def test_an_all_skip_run_still_backfills():
    """The regression this file exists for. A GPU-less worker re-offered an
    all-fallback dive returns only `NO_UPGRADE_AVAILABLE`, so there is nothing
    to persist -- but the project may still be pointed at no version at all,
    leaving every attached prediction invisible."""
    assert await _run([_skipped(1), _skipped(2)]) == DIVE
    assert not _PERSISTED, "nothing was improvable, so nothing should be written"
    assert _BACKFILLED == [DIVE], "the visibility repair must still run"

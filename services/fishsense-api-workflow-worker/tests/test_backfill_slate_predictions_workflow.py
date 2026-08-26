# pylint: disable=unused-argument
"""Workflow contract tests for the slate-prediction backfill.

Pins:
  1. BackfillSlatePredictionsWorkflow forwards its `dive_id` to the backfill
     activity and returns the attached-count verbatim.
  2. PredictSlateImagesParentWorkflow calls the backfill activity for the dive
     *after* persisting predictions, so already-populated dives surface the
     pre-annotations.
"""

from __future__ import annotations

import uuid
from typing import List, Tuple

import pytest
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_shared import (
    PredictSlateImage,
    PredictSlateImagesInput,
    SlatePredictionResult,
)
from fishsense_api_workflow_worker.activities.gpu_fallback import MODE_GPU
from fishsense_api_workflow_worker.workflows._dispatch import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
)
from fishsense_api_workflow_worker.workflows.backfill_slate_predictions_workflow import (  # noqa: E501  pylint: disable=line-too-long
    BackfillSlatePredictionsWorkflow,
)
from fishsense_api_workflow_worker.workflows.predict_slate_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictSlateImagesParentWorkflow,
)


@workflow.defn(name="PredictSlateImagesWorkflow")
class _StubSlateChild:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(
        self, payload: PredictSlateImagesInput
    ) -> List[SlatePredictionResult]:
        return [
            SlatePredictionResult(
                image_id=img.image_id,
                reference_points=[[1.0, 2.0]],
                confidence=0.9,
                rejected_reason=None,
                width=10,
                height=10,
            )
            for img in payload.images
        ]


@pytest.mark.asyncio
async def test_forwards_dive_id_and_returns_count():
    seen: List[int] = []

    @activity.defn(name="backfill_slate_predictions_for_dive_activity")
    async def backfill(dive_id: int) -> int:
        seen.append(dive_id)
        return 7

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-backfill-slate",
            workflows=[BackfillSlatePredictionsWorkflow],
            activities=[backfill],
        ):
            result = await env.client.execute_workflow(
                BackfillSlatePredictionsWorkflow.run,
                65,
                id=f"backfill-slate-{uuid.uuid4()}",
                task_queue="test-backfill-slate",
            )

    assert result == 7
    assert seen == [65]


def _predict_parent_stubs(calls: List[Tuple[str, int]]):
    @activity.defn(name="select_next_high_priority_dive_for_slate_prediction_activity")
    async def selector() -> int | None:
        return 65

    @activity.defn(name="resolve_slate_predict_inputs_activity")
    async def resolver(dive_id: int) -> PredictSlateImagesInput:
        return PredictSlateImagesInput(
            dive_id=dive_id,
            slate_id=11,
            slate_name="V-Slate 1",
            dpi=300.0,
            template_points=[[0.0, 0.0]],
            camera_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
            images=[PredictSlateImage(image_id=1, checksum="a")],
        )

    # The slate predict child runs on the GPU queue — "prefer a GPU", served by
    # the GPU Deployment or its CPU-only fallback.
    @activity.defn(name="ensure_gpu_worker_running_activity")
    async def ensure() -> str:
        return MODE_GPU

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stage_raw(dive_id: int) -> None:
        return None

    @activity.defn(name="stage_slate_pdf_activity")
    async def stage_pdf(slate_id: int) -> None:
        return None

    @activity.defn(name="persist_slate_predictions_activity")
    async def persist(results: list) -> None:
        calls.append(("persist", len(results)))

    @activity.defn(name="backfill_slate_predictions_for_dive_activity")
    async def backfill(dive_id: int) -> int:
        calls.append(("backfill", dive_id))
        return 0

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def cleanup(dive_id: int) -> None:
        return None

    return [selector, resolver, ensure, stage_raw, stage_pdf, persist, backfill, cleanup]


@pytest.mark.asyncio
async def test_predict_parent_calls_backfill_after_persist():
    calls: List[Tuple[str, int]] = []
    acts = _predict_parent_stubs(calls)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-predict-slate-parent",
            workflows=[PredictSlateImagesParentWorkflow],
            activities=acts,
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_GPU_TASK_QUEUE,
            workflows=[_StubSlateChild],
            activities=acts,
        ):
            result = await env.client.execute_workflow(
                PredictSlateImagesParentWorkflow.run,
                id=f"predict-slate-parent-{uuid.uuid4()}",
                task_queue="test-predict-slate-parent",
            )

    assert result == 65
    assert ("persist", 1) in calls
    assert ("backfill", 65) in calls
    assert calls.index(("persist", 1)) < calls.index(("backfill", 65))

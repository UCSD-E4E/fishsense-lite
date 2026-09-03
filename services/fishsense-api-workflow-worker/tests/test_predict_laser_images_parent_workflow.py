# pylint: disable=unused-argument
# `_stubs` and `_run` take one parameter per step the parent drives, so their
# argument counts track the workflow's command sequence by design; collapsing
# them into a config object would hide the very thing this test pins.
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""Workflow contract test for PredictLaserImagesParentWorkflow.

Pins down:
  1. Selector None -> parent returns None, no resolver/child.
  2. Full path: dispatches the child on the data-worker **GPU** queue with the
     deterministic id `predict-laser-{dive_id}`, then persists the child's
     results and returns the dive_id.
  3. Resolver returning 0 images skips the child + persist.
  4. No GPU *or* CPU-fallback worker available -> the parent bails before
     staging anything.

(2) and (4) are the halves of the GPU split. This is the only parent that
dispatches to `fishsense_data_processing_gpu_queue`; the stub child is
registered there and nowhere else, so a regression that sent it back to the CPU
queue would hang rather than pass.

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
    LaserAutoAcceptSummary,
    LaserPredictionResult,
    PredictLaserImage,
    PredictLaserImagesInput,
)
from fishsense_api_workflow_worker.activities.gpu_fallback import (
    MODE_CPU_FALLBACK,
    MODE_GPU,
    MODE_UNAVAILABLE,
)
from fishsense_api_workflow_worker.workflows._dispatch import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)
from fishsense_api_workflow_worker.workflows.predict_laser_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictLaserImagesParentWorkflow,
)

_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]

# Dive id the stub gate refuses (no consensus, nothing auto-accepted).
_REFUSED_DIVE_ID = 99


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


@workflow.defn(name="EvaluateLaserAutoAcceptWorkflow")
class _StubGate:
    # pylint: disable=too-few-public-methods
    """Stands in for the data-worker gate. Registered on the CPU queue only,
    so a regression that dispatched it to the GPU queue would hang rather
    than pass — the mirror of what `_StubChild` pins for the predict child."""

    @workflow.run
    async def run(self, dive_id: int) -> LaserAutoAcceptSummary:
        await workflow.execute_activity(
            "_record_gate_dispatch",
            args=(workflow.info().workflow_id, dive_id),
            schedule_to_close_timeout=timedelta(seconds=5),
        )
        # Dive `_REFUSED_DIVE_ID` stands for a gate that cleared nothing — a
        # dive whose predictions did not agree. Keyed on the id rather than on
        # module state so the stub stays deterministic under the workflow
        # sandbox and needs no second registration under the same name.
        refused = dive_id == _REFUSED_DIVE_ID
        return LaserAutoAcceptSummary(
            dive_id=dive_id,
            eligible=not refused,
            reason="weak_consensus" if refused else None,
            n_points=2,
            auto_accepted=0 if refused else 1,
            verdicts={"dive_ineligible": 2} if refused else {
                "auto_accepted": 1,
                "off_line": 1,
            },
        )


def _stubs(
    dive_id, images, child_ids, persisted, staged=None, mode=MODE_GPU,
    backfilled=None, gate_ids=None, applied=None,
):
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

    @activity.defn(name="ensure_gpu_worker_running_activity")
    async def ensure_gpu() -> str:
        return mode

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def stage(d: int) -> None:
        if staged is not None:
            staged.append(d)

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def cleanup(d: int) -> None:
        return None

    @activity.defn(name="persist_laser_predictions_activity")
    async def persist(results: list) -> int:
        persisted.extend(results)
        return len(results)

    sink = backfilled if backfilled is not None else []

    @activity.defn(name="backfill_laser_predictions_for_dive_activity")
    async def backfill(d: int) -> int:
        sink.append(d)
        return 0

    @activity.defn(name="_record_child_dispatch")
    async def record(workflow_id: str, d: int) -> None:
        child_ids.append(workflow_id)

    @activity.defn(name="ensure_data_worker_running_activity")
    async def ensure_cpu() -> None:
        return None

    gate_sink = gate_ids if gate_ids is not None else []

    @activity.defn(name="_record_gate_dispatch")
    async def record_gate(workflow_id: str, d: int) -> None:
        gate_sink.append(workflow_id)

    apply_sink = applied if applied is not None else []

    @activity.defn(name="apply_laser_auto_accept_for_dive_activity")
    async def apply_auto_accept(d: int) -> int:
        apply_sink.append(d)
        return len(apply_sink)

    return [
        selector,
        resolver,
        ensure_gpu,
        stage,
        cleanup,
        persist,
        backfill,
        record,
        ensure_cpu,
        record_gate,
        apply_auto_accept,
    ]


async def _run(
    env, dive_id, images, child_ids, persisted, staged=None, mode=MODE_GPU,
    backfilled=None, gate_ids=None, applied=None,
):
    activities = _stubs(
        dive_id, images, child_ids, persisted, staged, mode, backfilled, gate_ids,
        applied,
    )
    # Two workers: the parent (+ its activities) on the parent queue, and the
    # stub child on the data-processing GPU queue the parent dispatches to —
    # the activities are registered on both so the child's
    # _record_child_dispatch and the parent's persist both resolve.
    async with Worker(
        env.client,
        task_queue="test-predict-parent",
        workflows=[PredictLaserImagesParentWorkflow],
        activities=activities,
    ), Worker(
        env.client,
        task_queue=DATA_PROCESSING_GPU_TASK_QUEUE,
        workflows=[_StubChild],
        activities=activities,
    ), Worker(
        env.client,
        task_queue=DATA_PROCESSING_TASK_QUEUE,
        workflows=[_StubGate],
        activities=activities,
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
        result = await _run(env, None, [], child_ids, persisted)
    assert result is None
    assert not child_ids
    assert not persisted


@pytest.mark.asyncio
async def test_full_path_dispatches_child_and_persists():
    child_ids: List[str] = []
    persisted: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, 440, [(1, "a"), (2, "b")], child_ids, persisted
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
        result = await _run(env, 440, [], child_ids, persisted)
    assert result == 440
    assert not child_ids
    assert not persisted


@pytest.mark.asyncio
async def test_cpu_fallback_capacity_still_dispatches():
    """A CPU-fallback pod serves the same queue, so the parent's path is
    identical — the mode is informational, not routing."""
    child_ids: List[str] = []
    persisted: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, 441, [(1, "a")], child_ids, persisted, mode=MODE_CPU_FALLBACK
        )
    assert result == 441
    assert child_ids == ["predict-laser-441"]


@pytest.mark.asyncio
async def test_unavailable_capacity_bails_before_staging_anything():
    """No worker can serve the GPU queue. Staging the dive's raw `.ORF` bytes
    from the NAS and then dispatching would not fail — the child would sit
    Running until its execution timeout — so the parent must stop here and let
    the cohort selector re-offer the dive next hour."""
    child_ids: List[str] = []
    persisted: List = []
    staged: List[int] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, 442, [(1, "a")], child_ids, persisted, staged, MODE_UNAVAILABLE
        )
    assert result is None
    assert not staged
    assert not child_ids
    assert not persisted


@pytest.mark.asyncio
async def test_backfill_runs_after_persist_so_predictions_reach_labelers():
    """Persisting alone is invisible.

    Populate seeds a task's pre-annotation once, at import time, and dedupes by
    URL — so for a dive whose tasks already exist, a new `LaserPrediction`
    changes the database and nothing a labeler sees. That is every dive the
    re-prediction cohort selects, since it only picks dives still being
    labeled. Without this call the whole re-prediction path is busywork.
    """
    persisted: List = []
    backfilled: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env,
            dive_id=440,
            images=[(1, "a"), (2, "b")],
            child_ids=[],
            persisted=persisted,
            backfilled=backfilled,
        )
    assert result == 440
    assert persisted, "nothing persisted, so the backfill assertion is vacuous"
    assert backfilled == [440]


@pytest.mark.asyncio
async def test_backfill_is_skipped_when_the_child_returned_nothing():
    """No results means no prediction changed, so there is nothing to attach
    and no reason to spend Label Studio calls listing every project."""
    persisted: List = []
    backfilled: List = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        await _run(
            env,
            dive_id=441,
            images=[],
            child_ids=[],
            persisted=persisted,
            backfilled=backfilled,
        )
    assert not backfilled


@pytest.mark.asyncio
async def test_gate_runs_on_the_cpu_queue_with_a_deterministic_id():
    """The auto-accept gate is dispatched after the predictions are persisted,
    on the CPU queue.

    Queue matters and the failure is silent: `_StubGate` is registered ONLY on
    `DATA_PROCESSING_TASK_QUEUE`, so a regression sending it to the GPU queue
    would hang until the execution timeout rather than fail a assertion. The
    gate is a line fit — holding a contended NRP card through it is waste, and
    the CPU Deployment is what every other math stage uses.
    """
    child_ids: List[str] = []
    persisted: List = []
    gate_ids: List[str] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, 77, [(1, "c1"), (2, "c2")], child_ids, persisted,
            gate_ids=gate_ids,
        )
    assert result == 77
    assert gate_ids == ["auto-accept-laser-77"]


@pytest.mark.asyncio
async def test_gate_is_skipped_when_the_dive_had_nothing_to_predict():
    """No predictions means no new dots, so there is nothing to re-judge and
    the dive's existing verdicts still stand. Skipping also avoids waking the
    CPU Deployment for a no-op."""
    child_ids: List[str] = []
    persisted: List = []
    gate_ids: List[str] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(env, 78, [], child_ids, persisted, gate_ids=gate_ids)
    assert result == 78
    assert not gate_ids


@pytest.mark.asyncio
async def test_cleared_verdicts_are_applied_to_tasks_that_already_exist():
    """Populate imports a task once, so a dive still being labeled would get a
    verdict in the database and nothing a labeler sees. This is the step that
    closes that gap — the same one #493 closed for slate predictions."""
    child_ids: List[str] = []
    persisted: List = []
    gate_ids: List[str] = []
    applied: List[int] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, 77, [(1, "c1"), (2, "c2")], child_ids, persisted,
            gate_ids=gate_ids, applied=applied,
        )
    assert result == 77
    assert gate_ids == ["auto-accept-laser-77"]
    assert applied == [77]


@pytest.mark.asyncio
async def test_nothing_is_applied_when_the_gate_cleared_nothing():
    """A refused dive has no verdicts to apply. Skipping saves a pass over
    every task in Label Studio for a dive that, by definition, is the one whose
    predictions could not be trusted."""
    child_ids: List[str] = []
    persisted: List = []
    gate_ids: List[str] = []
    applied: List[int] = []
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env, _REFUSED_DIVE_ID, [(1, "c1")], child_ids, persisted,
            gate_ids=gate_ids, applied=applied,
        )
    assert result == _REFUSED_DIVE_ID
    assert gate_ids == [f"auto-accept-laser-{_REFUSED_DIVE_ID}"]
    assert not applied

"""Both parents that dispatch the gate child must wait the same amount.

`EvaluateLaserAutoAcceptWorkflow` has two callers -- the hourly backlog drain
and the inline path off the predict parent -- and they are the reason the
budget lives in `fishsense_shared` instead of being written twice. When the
gate's queue-wait tolerance was raised after the 2026-09-04 `ScheduleToStart`
failures, a literal left behind at either call site would have capped the
child below its own activity, so the fix would have appeared to work on one
path and silently not on the other.

The parent bodies are exercised directly with `_dispatch` stubbed out, so this
asserts the dispatch arguments without a Temporal environment.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fishsense_shared import (
    GATE_ACTIVITY_TIMEOUT,
    GATE_CHILD_EXECUTION_TIMEOUT,
    LaserAutoAcceptSummary,
    LaserPredictionResult,
    PredictLaserImage,
    PredictLaserImagesInput,
)

from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch
from fishsense_api_workflow_worker.workflows.evaluate_laser_auto_accept_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    EvaluateLaserAutoAcceptParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.predict_laser_images_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictLaserImagesParentWorkflow,
)

_DIVE_ID = 7

_SUMMARY = LaserAutoAcceptSummary(
    dive_id=_DIVE_ID,
    eligible=True,
    reason=None,
    n_points=30,
    auto_accepted=7,
    verdicts={"auto_accepted": 7},
)


def _stub_dispatch(monkeypatch) -> list[dict]:
    """Replace every `_dispatch` step with a stub, recording child dispatches.

    Deliberately stubs the whole module surface rather than only
    `dispatch_child`: the predict parent's body runs several steps before the
    gate, and any real one would need a Temporal context.
    """
    children: list[dict] = []

    async def _dispatch_child(workflow_name, inputs, **kwargs):
        children.append({"workflow_name": workflow_name, "inputs": inputs, **kwargs})
        if workflow_name == "EvaluateLaserAutoAcceptWorkflow":
            return _SUMMARY
        return [LaserPredictionResult(image_id=1, x=10.0, y=20.0, confidence=0.9)]

    async def _select_dive(_activity_name):
        return _DIVE_ID

    async def _resolve_inputs(_activity_name, dive_id, _result_type):
        return PredictLaserImagesInput(
            dive_id=dive_id,
            images=[PredictLaserImage(image_id=1, checksum="abc")],
            camera_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

    async def _noop(*_args, **_kwargs):
        return None

    async def _wake_gpu_worker():
        return "gpu"

    # `workflow.logger` reaches for the workflow event loop on every call, and
    # both parents log their verdict mix. A plain logger keeps the bodies
    # runnable outside a Temporal context.
    monkeypatch.setattr(workflow, "logger", logging.getLogger("test-workflow"))
    monkeypatch.setattr(_dispatch, "dispatch_child", _dispatch_child)
    monkeypatch.setattr(_dispatch, "select_dive", _select_dive)
    monkeypatch.setattr(_dispatch, "resolve_inputs", _resolve_inputs)
    monkeypatch.setattr(_dispatch, "wake_data_worker", _noop)
    monkeypatch.setattr(_dispatch, "wake_gpu_worker", _wake_gpu_worker)
    monkeypatch.setattr(_dispatch, "stage_raw", _noop)
    monkeypatch.setattr(_dispatch, "cleanup_raw", _noop)
    monkeypatch.setattr(_dispatch, "run_sdk_activity", _noop)
    return children


def _gate_dispatch(children: list[dict]) -> dict:
    gate = [c for c in children if c["workflow_name"] == "EvaluateLaserAutoAcceptWorkflow"]
    assert len(gate) == 1, f"expected exactly one gate dispatch, got {children}"
    return gate[0]


@pytest.fixture(name="drain_gate")
def _drain_gate(monkeypatch) -> dict:
    children = _stub_dispatch(monkeypatch)
    asyncio.run(EvaluateLaserAutoAcceptParentWorkflow().run())
    return _gate_dispatch(children)


@pytest.fixture(name="predict_gate")
def _predict_gate(monkeypatch) -> dict:
    children = _stub_dispatch(monkeypatch)
    asyncio.run(PredictLaserImagesParentWorkflow().run())
    return _gate_dispatch(children)


def test_the_drain_waits_the_shared_budget(drain_gate):
    assert drain_gate["execution_timeout"] == GATE_CHILD_EXECUTION_TIMEOUT


def test_the_predict_path_waits_the_shared_budget(predict_gate):
    assert predict_gate["execution_timeout"] == GATE_CHILD_EXECUTION_TIMEOUT


def test_both_paths_reuse_the_same_child_id(drain_gate, predict_gate):
    """The id is what makes a dive's gate run exclusive of itself across the
    two paths -- the drain firing at +22 and the predict parent's inline gate
    at +10 can otherwise judge the same dive concurrently."""
    assert drain_gate["child_id"] == predict_gate["child_id"]
    assert drain_gate["child_id"] == f"auto-accept-laser-{_DIVE_ID}"


def test_the_child_outlasts_its_own_activity(drain_gate):
    """A child that timed out before its activity would report an opaque
    `ChildWorkflowError`, hiding which bound was hit. Asserted at the call
    site, not only on the constants, so a stray literal here cannot reintroduce
    the inversion."""
    assert drain_gate["execution_timeout"] > GATE_ACTIVITY_TIMEOUT

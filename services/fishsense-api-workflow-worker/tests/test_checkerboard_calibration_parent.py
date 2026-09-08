"""Workflow contract test for PerformCheckerboardCalibrationParentWorkflow.

Two properties, both about not spending a NAS transfer for nothing:

  * bytes are staged only after the resolver has found frames;
  * the staged scratch is cleaned up even when the fit fails — which, for this
    stage, is the *expected* shape of "these frames hold no detectable board",
    not an exceptional one.

The resolver's own predicate lives in `test_checkerboard_calibration_resolver.py`;
it is a separate module because it needs numpy, and a test module that both
imports numpy and defines a `@workflow.defn` breaks Temporal's workflow
sandbox (it re-imports the module, and a C extension cannot load twice).
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fishsense_shared import (
    CheckerboardCalibrationImage,
    PerformCheckerboardCalibrationInput,
)
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows._dispatch import (
    DATA_PROCESSING_TASK_QUEUE,
)
from fishsense_api_workflow_worker.workflows.perform_checkerboard_calibration_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PerformCheckerboardCalibrationParentWorkflow,
)

_K = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]

#: What ran, in order. Module level because Temporal refuses `@workflow.run`
#: on a local class, so the stubs cannot close over per-test state.
_CALLS: list[str] = []
_CHILD_FAILS = False


@pytest.fixture(autouse=True)
def _reset():
    _CALLS.clear()
    global _CHILD_FAILS  # pylint: disable=global-statement
    _CHILD_FAILS = False
    yield
    _CHILD_FAILS = False


def _payload(frames: int) -> PerformCheckerboardCalibrationInput:
    return PerformCheckerboardCalibrationInput(
        dive_id=488,
        camera_id=9,
        camera_matrix=_K,
        distortion_coefficients=[0.0] * 5,
        target_rows=10,
        target_cols=14,
        square_size_m=0.0254,
        images=[
            CheckerboardCalibrationImage(
                image_id=100 + n, checksum=f"{n:032d}", laser_x=600.0, laser_y=500.0
            )
            for n in range(frames)
        ],
    )


@workflow.defn(name="PerformCheckerboardCalibrationWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    """Records that it ran, through an activity.

    Through an activity, not by appending to `_CALLS` directly: the workflow
    sandbox re-imports this module, so workflow code mutates its own copy of
    the list. Activities run unsandboxed and see the real one — which is also
    why the failure is raised there.
    """

    @workflow.run
    async def run(self, payload: PerformCheckerboardCalibrationInput) -> int:
        await workflow.execute_activity(
            "_record_child_dispatch",
            args=(payload.dive_id, len(payload.images)),
            schedule_to_close_timeout=timedelta(seconds=5),
            # One attempt: the failure case here is a *permanent* one ("no
            # board in these frames"), and the default policy would retry it
            # until the test's schedule-to-close.
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        return 31


def _stub_activities(*, dive_id: int | None, frames: int):
    @activity.defn(
        name="select_next_high_priority_dive_for_checkerboard_calibration_activity"
    )
    async def _select() -> int | None:
        _CALLS.append("select")
        return dive_id

    @activity.defn(name="resolve_checkerboard_calibration_inputs_activity")
    async def _resolve(_dive_id: int) -> PerformCheckerboardCalibrationInput:
        _CALLS.append("resolve")
        return _payload(frames)

    @activity.defn(name="ensure_data_worker_running_activity")
    async def _wake() -> None:
        _CALLS.append("wake")

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def _stage(_dive_id: int) -> None:
        _CALLS.append("stage")

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def _cleanup(_dive_id: int) -> None:
        _CALLS.append("cleanup")

    return [_select, _resolve, _wake, _stage, _cleanup]


@activity.defn(name="_record_child_dispatch")
async def _record_child_dispatch(_dive_id: int, _frames: int) -> None:
    _CALLS.append("child")
    if _CHILD_FAILS:
        raise ValueError("insufficient checkerboard laser points (0 < 2)")


async def _run_parent(
    task_queue: str, *, dive_id=488, frames=2, child_already_running=False
):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[PerformCheckerboardCalibrationParentWorkflow],
            activities=_stub_activities(dive_id=dive_id, frames=frames),
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
            activities=[_record_child_dispatch],
        ):
            if child_already_running:
                # Occupy the deterministic child id, which is what makes the
                # real dispatch raise `WorkflowAlreadyStartedError` and return
                # the sentinel. Started on a queue NOBODY serves so it stays
                # Running for the whole test — a workflow id is claimed
                # namespace-wide, so it blocks the parent's dispatch regardless
                # of which queue holds it.
                await env.client.start_workflow(
                    _StubChildWorkflow.run,
                    _payload(frames),
                    id=f"perform-checkerboard-calibration-{dive_id}",
                    task_queue="no-worker-serves-this-queue",
                )
            return await env.client.execute_workflow(
                PerformCheckerboardCalibrationParentWorkflow.run,
                id=f"wf-{task_queue}",
                task_queue=task_queue,
            )


@pytest.mark.asyncio
async def test_parent_stages_dispatches_and_cleans_up():
    assert await _run_parent("test-checkerboard-happy") == 488
    assert _CALLS == [
        "select",
        "resolve",
        "wake",
        "stage",
        "wake",
        "child",
        "cleanup",
    ]


@pytest.mark.asyncio
async def test_parent_wakes_the_worker_again_after_staging():
    """Staging outlasts the scale-to-zero sweeper, so one early wake is not enough.

    This parent stages a dive's raw `.ORF`s before dispatching — ~30 minutes
    for a 133-frame folder over the ~1 MB/s NAS link — and staging runs on the
    *api-worker's* queue, so `fishsense_data_processing_queue` has nothing
    Running the whole time. The +55 sweeper therefore sees an idle queue and
    scales the CPU worker to 0, and the child then lands on an unserved queue:
    it does not fail, it hangs until its 2h execution timeout, and SKIP overlap
    suppresses the next firings behind it.

    It is likeliest exactly when these dives get their turn — the sweeper only
    scales down once the preprocess backlog has drained, which is when nothing
    else is keeping the queue warm.

    The wake is idempotent (an absolute target, never an increment), so calling
    it twice costs one cheap activity and closes the window however long
    staging takes. The early one is kept so the pod's cold start still overlaps
    staging.
    """
    await _run_parent("test-checkerboard-double-wake")

    assert _CALLS.count("wake") == 2
    # The second one is what matters: nothing may sit between it and the child.
    assert _CALLS.index("child") == _CALLS.index("wake", _CALLS.index("stage")) + 1


@pytest.mark.asyncio
async def test_parent_does_nothing_when_the_cohort_is_empty():
    assert await _run_parent("test-checkerboard-empty", dive_id=None) is None
    assert _CALLS == ["select"]


@pytest.mark.asyncio
async def test_parent_does_not_stage_when_the_resolver_finds_no_frames():
    """Staging a dive's raw bytes over a ~1 MB/s NAS link to dispatch nothing
    is the expensive way to discover a selector/resolver disagreement."""
    assert await _run_parent("test-checkerboard-no-frames", frames=0) == 488
    assert _CALLS == ["select", "resolve"]


@pytest.mark.asyncio
async def test_parent_leaves_the_scratch_alone_when_a_child_already_owns_it():
    """Another run's child is still reading those `.ORF`s.

    `dispatch_child` returns `CHILD_ALREADY_RUNNING` when a prior child with
    this dive's id is still running — a manual run overlapping the schedule.
    Cleaning up then deletes the staged raw bytes out from under it, which is
    the prod dive 442 incident (2026-09-07): a manual stage-0.1 parent's child
    had scheduled all 259 per-image activities when the next scheduled firing
    swept its scratch away.

    This parent's cleanup is in a `finally`, so it is *more* exposed to that
    than the preprocess parents were — it would clean up on the
    already-running path too.
    """
    await _run_parent("test-checkerboard-already-running", child_already_running=True)

    assert "cleanup" not in _CALLS


@pytest.mark.asyncio
async def test_parent_cleans_up_even_when_the_fit_fails():
    """A failing fit is the expected shape of "no board in these frames".

    The dive stays in the cohort and is re-selected hourly until an operator
    clears its calibration target or parks it, so leaving a dive's worth of
    `.ORF` scratch behind on every firing would accumulate for exactly the
    dives that are already stuck. The preprocess parents do not do this;
    for them a child failure is unusual rather than expected.
    """
    global _CHILD_FAILS  # pylint: disable=global-statement
    _CHILD_FAILS = True

    with pytest.raises(WorkflowFailureError):
        await _run_parent("test-checkerboard-child-fails")

    assert "cleanup" in _CALLS
    assert _CALLS.index("cleanup") > _CALLS.index("child")

"""Workflow contract tests for VerifyCheckerboardLatticeParentWorkflow.

Three properties, each of which cost a review finding on 2026-09-11 and each
of which fails expensively — after every dive has been staged from the NAS and
rendered, which is the slow part:

* **the import is chunked.** All dives' renders used to go to one activity
  argument. A 10x14 render measures ~2.6 KB, so this workflow's own documented
  example is comfortably fine and an uncapped run over the same dives is not —
  past Temporal's 2 MB blob limit, discovered only at the very end.
* **one bad dive does not discard the rest.** The import is deferred to the
  end for blinding, so without isolation a single unusable `dive_id` throws
  away every dive rendered before it.
* **chunking happens after the shuffle.** Chunking first would make each chunk
  a single dive's frames, restoring exactly the per-dive ordering the shuffle
  exists to destroy.
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from fishsense_shared import (
    CheckerboardCalibrationImage,
    CheckerboardLatticeRender,
    PerformCheckerboardCalibrationInput,
    VerifyCheckerboardLatticeInput,
)
from temporalio import activity, workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_api_workflow_worker.workflows._dispatch import (
    DATA_PROCESSING_TASK_QUEUE,
)
from fishsense_api_workflow_worker.workflows.verify_checkerboard_lattice_parent_workflow import (  # noqa: E501  pylint: disable=line-too-long
    VerifyCheckerboardLatticeParentInput,
    VerifyCheckerboardLatticeParentWorkflow,
)

_K = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]

#: Module level for the same reason the calibration parent's test does it: the
#: workflow sandbox re-imports this module, so only activities see the real
#: lists.
_IMPORT_CHUNKS: list[list[int]] = []
_RESOLVE_FAILS: set[int] = set()
_FRAMES_PER_DIVE = 0


@pytest.fixture(autouse=True)
def _reset():
    _IMPORT_CHUNKS.clear()
    _RESOLVE_FAILS.clear()
    yield
    _IMPORT_CHUNKS.clear()
    _RESOLVE_FAILS.clear()


def _resolved(dive_id: int, frames: int) -> PerformCheckerboardCalibrationInput:
    return PerformCheckerboardCalibrationInput(
        dive_id=dive_id,
        camera_id=9,
        camera_matrix=_K,
        distortion_coefficients=[0.0] * 5,
        target_rows=10,
        target_cols=14,
        square_size_m=0.042,
        images=[
            CheckerboardCalibrationImage(
                image_id=dive_id * 1000 + n,
                checksum=f"{dive_id:04d}{n:028d}",
                laser_x=600.0,
                laser_y=500.0,
            )
            for n in range(frames)
        ],
    )


@workflow.defn(name="VerifyCheckerboardLatticeWorkflow")
class _StubChildWorkflow:
    # pylint: disable=too-few-public-methods
    """Returns one render per image it was dispatched with.

    The payload annotation is load-bearing, not documentation: Temporal reads
    the run method's type hints to decide what to deserialise the argument
    into. Without it the workflow receives a plain dict and every attribute
    access below fails.
    """

    @workflow.run
    async def run(self, payload: VerifyCheckerboardLatticeInput) -> list:
        images = payload.images
        if payload.sample_limit is not None:
            images = images[: payload.sample_limit]
        return [
            CheckerboardLatticeRender(
                image_id=image.image_id,
                checksum=image.checksum,
                detected_rows=10,
                detected_cols=14,
                median_spacing_px=32.0,
                corners=[[1.0, 2.0]] * 4,
                width=4000,
                height=3000,
            )
            for image in images
        ]


def _stub_activities(frames: int):
    @activity.defn(name="resolve_checkerboard_calibration_inputs_activity")
    async def _resolve(dive_id: int) -> PerformCheckerboardCalibrationInput:
        if dive_id in _RESOLVE_FAILS:
            raise ValueError(f"dive_id={dive_id} has no calibration target")
        return _resolved(dive_id, frames)

    @activity.defn(name="ensure_data_worker_running_activity")
    async def _wake() -> None:
        return None

    @activity.defn(name="stage_raw_bytes_for_dive_activity")
    async def _stage(_dive_id: int) -> None:
        return None

    @activity.defn(name="cleanup_raw_bytes_for_dive_activity")
    async def _cleanup(_dive_id: int) -> None:
        return None

    @activity.defn(name="create_checkerboard_lattice_label_studio_project_activity")
    async def _create() -> int:
        return 4242

    @activity.defn(name="populate_checkerboard_lattice_label_studio_project_activity")
    async def _populate(_project_id: int, renders: list) -> int:
        _IMPORT_CHUNKS.append(
            [
                r["image_id"] if isinstance(r, dict) else r.image_id
                for r in renders
            ]
        )
        return len(renders)

    return [_resolve, _wake, _stage, _cleanup, _create, _populate]


async def _run(dive_ids, *, frames=3, sample_limit=None, workflow_id="wf-lattice"):
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="lattice-parent-test",
            workflows=[VerifyCheckerboardLatticeParentWorkflow],
            activities=_stub_activities(frames),
        ), Worker(
            env.client,
            task_queue=DATA_PROCESSING_TASK_QUEUE,
            workflows=[_StubChildWorkflow],
        ):
            return await env.client.execute_workflow(
                VerifyCheckerboardLatticeParentWorkflow.run,
                VerifyCheckerboardLatticeParentInput(
                    dive_ids=dive_ids, sample_limit=sample_limit
                ),
                id=workflow_id,
                task_queue="lattice-parent-test",
                execution_timeout=timedelta(minutes=5),
            )


@pytest.mark.asyncio
async def test_imports_every_render_across_every_dive():
    imported = await _run([1, 2, 3], frames=4, workflow_id="wf-lattice-all")

    assert imported == 12
    assert sum(len(chunk) for chunk in _IMPORT_CHUNKS) == 12


@pytest.mark.asyncio
async def test_the_import_is_chunked_rather_than_one_payload():
    """250 renders must not travel as a single activity argument."""
    await _run([1, 2, 3, 4, 5], frames=50, workflow_id="wf-lattice-chunked")

    assert len(_IMPORT_CHUNKS) > 1
    assert all(len(chunk) <= 100 for chunk in _IMPORT_CHUNKS)


@pytest.mark.asyncio
async def test_chunks_are_shuffled_across_dives_not_grouped_by_dive():
    """Chunking before the shuffle would restore per-dive ordering.

    Label Studio serves tasks in import order, so a chunk holding one dive's
    frames hands a labeler that dive as a contiguous block — which is exactly
    what the shuffle exists to prevent. Image ids are `dive_id * 1000 + n`, so
    a chunk's dive membership is readable straight off them.
    """
    await _run([1, 2, 3, 4, 5], frames=50, workflow_id="wf-lattice-mixed")

    first_chunk_dives = {image_id // 1000 for image_id in _IMPORT_CHUNKS[0]}
    assert len(first_chunk_dives) > 1


@pytest.mark.asyncio
async def test_one_failing_dive_does_not_discard_the_others():
    """The expensive failure: staging is done by the time this could bite."""
    _RESOLVE_FAILS.add(2)

    imported = await _run([1, 2, 3], frames=4, workflow_id="wf-lattice-partial")

    assert imported == 8
    imported_dives = {
        image_id // 1000 for chunk in _IMPORT_CHUNKS for image_id in chunk
    }
    assert imported_dives == {1, 3}


@pytest.mark.asyncio
async def test_every_dive_failing_imports_nothing_and_does_not_raise():
    """A study of nothing is a reportable outcome, not a crash."""
    _RESOLVE_FAILS.update({1, 2})

    imported = await _run([1, 2], frames=4, workflow_id="wf-lattice-all-fail")

    assert imported == 0
    assert _IMPORT_CHUNKS == []


@pytest.mark.asyncio
async def test_sample_limit_reaches_the_child():
    imported = await _run(
        [1, 2], frames=10, sample_limit=3, workflow_id="wf-lattice-capped"
    )

    assert imported == 6

"""Workflow contract tests for checkerboard lattice verification.

The drawing is covered by `test_lattice_overlay.py` and the detector by
`test_checkerboard_detection.py`. What is left is the wiring, and two pieces of
it carry real consequence:

* **the board geometry travels per frame from the dispatch**, exactly as the
  calibration child's does. The whole study is about a mis-resolved pitch, so a
  verification run that rendered at a *different* pitch than the fit used would
  be answering a question nobody asked.
* **`sample_limit` is a stable head-of-list take.** A random sample would make
  two runs of the same study describe different frames, and a labeler's
  verdicts would stop being comparable to the run that produced them.
"""

from __future__ import annotations

import pytest
from fishsense_shared import (
    CheckerboardCalibrationImage,
    CheckerboardLatticeRender,
    VerifyCheckerboardLatticeInput,
)
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.workflows.verify_checkerboard_lattice_workflow import (  # noqa: E501  pylint: disable=line-too-long
    RenderCheckerboardLatticeInput,
    VerifyCheckerboardLatticeWorkflow,
)

CAMERA_MATRIX = [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]
SQUARE_SIZE_M = 0.042
TASK_QUEUE = "test-lattice-verification"


def _input(*, images=3, **overrides):
    kwargs = {
        "dive_id": 522,
        "camera_matrix": CAMERA_MATRIX,
        "distortion_coefficients": [0.0] * 5,
        "target_rows": 10,
        "target_cols": 14,
        "square_size_m": SQUARE_SIZE_M,
        "images": [
            CheckerboardCalibrationImage(
                image_id=200 + n,
                checksum=f"{n:032d}",
                laser_x=600.0 + n,
                laser_y=500.0,
            )
            for n in range(images)
        ],
    }
    kwargs.update(overrides)
    return VerifyCheckerboardLatticeInput(**kwargs)


def _rendered(payload: RenderCheckerboardLatticeInput) -> CheckerboardLatticeRender:
    return CheckerboardLatticeRender(
        image_id=payload.image_id,
        checksum=payload.checksum,
        detected_rows=10,
        detected_cols=14,
        median_spacing_px=32.0,
        corners=[[0.0, 0.0]] * (10 * 14),
        width=4000,
        height=3000,
    )


async def _run(payload, *, workflow_id, on_render=None):
    seen: list[RenderCheckerboardLatticeInput] = []

    @activity.defn(name="render_checkerboard_lattice")
    async def _render(inner: RenderCheckerboardLatticeInput):
        seen.append(inner)
        return (on_render or _rendered)(inner)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=TASK_QUEUE,
            workflows=[VerifyCheckerboardLatticeWorkflow],
            activities=[_render],
        ):
            result = await env.client.execute_workflow(
                VerifyCheckerboardLatticeWorkflow.run,
                payload,
                id=workflow_id,
                task_queue=TASK_QUEUE,
            )
    return result, seen


@pytest.mark.asyncio
async def test_every_frame_is_dispatched_with_the_board_geometry():
    result, seen = await _run(_input(images=3), workflow_id="wf-lattice-geometry")

    # Results keep input order — `asyncio.gather` returns in the order it was
    # given, whatever order the activities completed in.
    assert [r.image_id for r in result] == [200, 201, 202]
    # Dispatch order is not completion order under gather, so this is a set:
    # what must hold is that every frame was dispatched exactly once, with the
    # dive's board geometry.
    assert sorted(p.image_id for p in seen) == [200, 201, 202]
    assert {(p.target_rows, p.target_cols) for p in seen} == {(10, 14)}
    assert {p.square_size_m for p in seen} == {SQUARE_SIZE_M}


@pytest.mark.asyncio
async def test_each_frame_carries_its_own_dot():
    """The dot decides admission via `point_in_laser_region`, so a frame
    rendered against another frame's dot would admit the wrong population."""
    _, seen = await _run(_input(images=3), workflow_id="wf-lattice-dots")

    assert {p.image_id: p.laser_x for p in seen} == {200: 600.0, 201: 601.0, 202: 602.0}


@pytest.mark.asyncio
async def test_sample_limit_caps_the_frames_rendered():
    result, seen = await _run(
        _input(images=10, sample_limit=4), workflow_id="wf-lattice-cap"
    )

    assert len(seen) == 4
    assert len(result) == 4


@pytest.mark.asyncio
async def test_sample_limit_is_a_stable_head_of_list_take():
    """Two runs of the same study must render the same frames.

    Determinism is forced on workflow code anyway, but the property worth
    pinning is the *choice*: a seeded random sample would also replay, and
    would still make a second run describe a different set of frames than the
    verdicts already collected refer to.
    """
    _, first = await _run(
        _input(images=10, sample_limit=3), workflow_id="wf-lattice-stable-1"
    )
    _, second = await _run(
        _input(images=10, sample_limit=3), workflow_id="wf-lattice-stable-2"
    )

    # Sorted, because gather makes dispatch order arbitrary. The property under
    # test is *which* frames were chosen, not the order they went out in.
    assert sorted(p.image_id for p in first) == [200, 201, 202]
    assert sorted(p.image_id for p in first) == sorted(p.image_id for p in second)


@pytest.mark.asyncio
async def test_no_sample_limit_renders_every_frame():
    _, seen = await _run(_input(images=6), workflow_id="wf-lattice-uncapped")

    assert len(seen) == 6


@pytest.mark.asyncio
async def test_a_sample_limit_above_the_frame_count_is_harmless():
    _, seen = await _run(
        _input(images=2, sample_limit=50), workflow_id="wf-lattice-over-cap"
    )

    assert len(seen) == 2


@pytest.mark.asyncio
async def test_frames_with_no_lattice_are_returned_not_dropped():
    """A skipped frame still comes back, so the parent can tally *why*.

    The api-worker turns only rendered frames into tasks, but the skip reasons
    are the other half of the finding: a dive whose boards mostly fail to
    detect is telling you something different from one whose lattice is wrong,
    and the two are indistinguishable if the skips never leave this workflow.
    """

    def _skip(payload):
        if payload.image_id == 201:
            return CheckerboardLatticeRender(
                image_id=payload.image_id,
                checksum=payload.checksum,
                skip_reason="no_usable_board",
            )
        return _rendered(payload)

    result, _ = await _run(
        _input(images=3), workflow_id="wf-lattice-skips", on_render=_skip
    )

    assert len(result) == 3
    assert [r.skip_reason for r in result] == [None, "no_usable_board", None]


@pytest.mark.asyncio
async def test_renders_carry_the_frame_dimensions_back():
    """Label Studio stores keypoints as percentages, so a corner pixel is
    meaningless without the rectified frame it was measured in."""
    result, _ = await _run(_input(images=1), workflow_id="wf-lattice-dims")

    assert (result[0].width, result[0].height) == (4000, 3000)

"""Fan out lattice rendering across a dive's calibration frames.

The data-worker child of `VerifyCheckerboardLatticeParentWorkflow`. One phase,
unlike its calibration sibling: there is nothing to fit afterwards, because
this stage answers a question about the detector rather than producing a
number. Every frame is independent and the workflow just collects the renders
so the api-worker can turn them into Label Studio tasks.

Runs on the **CPU** queue. Each frame is a rawpy decode peaking at 1-3 GB,
which is the memory ceiling behind that worker's `max_concurrent_activities`
of 2 — the same reason the calibration child lives there rather than on the
light queue.

`sample_limit` is applied here rather than by the parent's resolver, and
deliberately: the cap is about how much *rendering and labeling* to do, and the
parent has already staged the dive's raw bytes by the time this runs. Trimming
in the resolver would also mean staging fewer frames, which sounds like a
saving until a later run wants a different sample and has to re-stage from NAS.
"""

import asyncio
from datetime import timedelta
from typing import List

from fishsense_shared import (
    CheckerboardLatticeRender,
    VerifyCheckerboardLatticeInput,
)
from pydantic import BaseModel
from temporalio import workflow

__all__ = [
    "RenderCheckerboardLatticeInput",
    "VerifyCheckerboardLatticeWorkflow",
]


class RenderCheckerboardLatticeInput(BaseModel):
    """Per-image payload for the `render_checkerboard_lattice` activity."""

    image_id: int
    checksum: str
    laser_x: float
    laser_y: float
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    # The declared board's INTERIOR corners — an upper bound on what may be
    # detected, not the grid asked for. See `checkerboard_detection`.
    target_rows: int
    target_cols: int
    square_size_m: float


@workflow.defn
class VerifyCheckerboardLatticeWorkflow:
    # pylint: disable=too-few-public-methods
    """Render each sampled frame's lattice and return what was drawn."""

    @workflow.run
    async def run(
        self, payload: VerifyCheckerboardLatticeInput
    ) -> List[CheckerboardLatticeRender]:
        images = payload.images
        if payload.sample_limit is not None:
            # A plain head-of-list take, NOT a random sample. Workflow code has
            # to be deterministic to replay, so `random` is unavailable here
            # without Temporal's seeded side-effect machinery — and a stable
            # subset is worth more than a random one anyway: re-running the
            # study renders the same frames, so a labeler's verdicts stay
            # comparable across runs instead of quietly describing a different
            # sample.
            images = images[: payload.sample_limit]

        workflow.logger.info(
            "lattice verification dive_id=%d frames=%d of %d board=%dx%d",
            payload.dive_id,
            len(images),
            len(payload.images),
            payload.target_rows,
            payload.target_cols,
        )

        # `asyncio.gather`, the same shape as the calibration child. An earlier
        # version ran these sequentially, reasoning that the decode's memory
        # peak made parallel dispatch pointless. That was wrong twice over: the
        # worker's own `max_concurrent_activities = 2` already bounds how many
        # decodes run at once however many are dispatched, so the sequential
        # loop bought no safety — and it serialised the whole fan-out under one
        # execution timeout, so an uncapped dive could time out where its
        # calibration sibling, doing the same work on the same frames, would
        # not.
        renders = await asyncio.gather(
            *[
                workflow.execute_activity(
                    "render_checkerboard_lattice",
                    RenderCheckerboardLatticeInput(
                        image_id=image.image_id,
                        checksum=image.checksum,
                        laser_x=image.laser_x,
                        laser_y=image.laser_y,
                        camera_matrix=payload.camera_matrix,
                        distortion_coefficients=payload.distortion_coefficients,
                        target_rows=payload.target_rows,
                        target_cols=payload.target_cols,
                        square_size_m=payload.square_size_m,
                    ),
                    start_to_close_timeout=timedelta(minutes=10),
                    result_type=CheckerboardLatticeRender,
                )
                for image in images
            ]
        )

        return list(renders)

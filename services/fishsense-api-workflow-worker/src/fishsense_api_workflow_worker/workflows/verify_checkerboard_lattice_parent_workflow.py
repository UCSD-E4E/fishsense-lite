"""Checkerboard lattice-verification parent workflow (api-worker side).

Renders the lattices a set of calibrations were fitted from, and puts them in
front of a human in one shuffled Label Studio project.

**Operator-driven, with an explicit dive list, and no selector.** Every other
parent here drains a cohort: it asks the API what needs work and takes one
dive. This one is a study. Which dives belong in it is a question about the
*experiment* — which calibrations are suspect and which are the controls — and
no predicate over the database encodes that. Hard-coding one would also make
the population drift as the data changes, so two runs of the same study would
not be comparable.

**It exists because nothing automatic can answer the question.** A uniformly
mis-latticed detection is still a perfect grid, so it is an exact homography of
the modelled grid: `grid_residual_px` reads ~0, `_fits_declared_board` sees a
grid *smaller* than the board, and `check_fit_self_consistency` compares 2-D
dots that a pure scale error does not move. The board is then placed at a
fraction of its true distance and every depth, the fitted baseline and every
length scales with it. Measured 2026-09-11 against the known-length targets,
five of fourteen checkerboard calibrations carry a baseline that is a simple
multiple of the consensus 10.4 cm — 4.00x, 2.16x, 2.12x, 1.51x, 0.47x. That is
the signature of this fault and not of noise, but it is an inference from the
lengths, not an observation of the lattice. This stage makes the observation.

**Dives are rendered one at a time, and imported once at the end.** Import is
deferred so the tasks can be interleaved: a labeler who works one calibration's
frames in a block can see the block, and a project ordered by dive tells them
which group is which long before they have judged it. The verdicts would then
be about the grouping rather than the lattice.

**Not scheduled.** Run it by hand:

```
temporal workflow start \\
    --task-queue fishsense_api_queue \\
    --type VerifyCheckerboardLatticeParentWorkflow \\
    --workflow-id verify-lattice-<run-tag> \\
    --input '{"dive_ids": [493,495,496,499,501,507,509,518,521,522],
              "sample_limit": 20}'
```
"""

from datetime import timedelta
from typing import List, Optional

from fishsense_shared import (
    CheckerboardLatticeRender,
    PerformCheckerboardCalibrationInput,
    VerifyCheckerboardLatticeInput,
)
from pydantic import BaseModel
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch

__all__ = [
    "VerifyCheckerboardLatticeParentInput",
    "VerifyCheckerboardLatticeParentWorkflow",
]

#: Renders per import activity call. A 10x14 render measures ~6 KB, so 100 is
#: ~600 KB against Temporal's 2 MB payload limit — room for a denser board
#: without another look at this number, and small enough that one failed chunk
#: re-imports little. The dedupe-by-URL inside the import makes a retried chunk
#: a no-op.
_IMPORT_CHUNK_SIZE = 100


class VerifyCheckerboardLatticeParentInput(BaseModel):
    """Which calibrations to study, and how many frames of each.

    `dive_ids` are the dives whose *own* calibration frames are rendered, so
    they are calibration sources rather than the dives that borrow from them.
    Include controls: a run holding only the suspect calibrations cannot
    distinguish "this detector mis-latticed these five" from "this detector
    mis-lattices everything", and the second reading would condemn the good
    fits too.
    """

    dive_ids: List[int]
    # Twenty per calibration settles a systematic fault comfortably while
    # keeping the whole study to a couple of hundred frames of human work.
    # Rendering every frame of the ten dives under study would be ~976 tasks
    # to answer the same question.
    #
    # `None` lifts the cap, and there is a ceiling above it worth knowing:
    # the child returns every render in ONE Temporal payload, and at ~2.6 KB
    # per 10x14 render the largest dive in this corpus (522, at 256 dotted
    # frames) is ~0.66 MB against a 2 MB blob limit. A denser board or a much
    # larger dive would need the child to chunk its result too — the import
    # side already does, via `_IMPORT_CHUNK_SIZE`.
    sample_limit: Optional[int] = 20


@workflow.defn
class VerifyCheckerboardLatticeParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Render each dive's lattices, then import them all as one shuffled project.

    Returns the number of Label Studio tasks imported.
    """

    @workflow.run
    async def run(self, payload: VerifyCheckerboardLatticeParentInput) -> int:
        renders: List[CheckerboardLatticeRender] = []
        failed: List[int] = []
        for dive_id in payload.dive_ids:
            # Per-dive isolation. Without it one unusable dive_id — a missing
            # `calibration_target_id`, absent intrinsics, a resolver raising
            # under the fail-fast retry policy — discards every dive rendered
            # before it, along with all of their NAS staging. The import is
            # deferred to the end for blinding, so there is a long window in
            # which that loss is total and silent.
            #
            # A dive that fails is reported and skipped: a study of nine
            # calibrations is still a study, while a study of none is an hour
            # of staging thrown away.
            try:
                renders.extend(await self._render_dive(dive_id, payload.sample_limit))
            except Exception as exc:  # pylint: disable=broad-except
                failed.append(dive_id)
                workflow.logger.error(
                    "lattice verification failed for dive_id=%d: %s", dive_id, exc
                )

        if failed:
            workflow.logger.error(
                "lattice verification skipped %d of %d dives: %s",
                len(failed),
                len(payload.dive_ids),
                failed,
            )

        if not renders:
            workflow.logger.warning("lattice verification produced no renders")
            return 0

        # `workflow.random()` is Temporal's seeded, replay-safe RNG — the stdlib
        # one would make this workflow non-deterministic and break replay. The
        # shuffle is what keeps the study blind: without it the project is
        # ordered by dive, and Label Studio serves tasks in import order.
        workflow.random().shuffle(renders)

        project_id = await workflow.execute_activity(
            "create_checkerboard_lattice_label_studio_project_activity",
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        # Imported in chunks, because the whole set does not fit in one
        # Temporal payload. A 10x14 render carries 140 corner pairs and
        # measures ~6 KB, so this workflow's own documented example (10 dives
        # x 20 frames) is ~1.2 MB and an uncapped run over the same dives is
        # ~8 MB — past the 2 MB blob limit, and it would fail *after* every
        # dive had been staged from the NAS and rendered.
        #
        # Chunking after the shuffle, never before: the shuffle is what
        # interleaves the dives, so each chunk is already a mixed slice and the
        # import order a labeler sees stays blind.
        imported = 0
        for start in range(0, len(renders), _IMPORT_CHUNK_SIZE):
            chunk = renders[start : start + _IMPORT_CHUNK_SIZE]
            imported += await workflow.execute_activity(
                "populate_checkerboard_lattice_label_studio_project_activity",
                args=(project_id, chunk),
                schedule_to_close_timeout=timedelta(minutes=30),
            )

        workflow.logger.info(
            "lattice verification imported %d tasks from %d of %d dives",
            imported,
            len(payload.dive_ids) - len(failed),
            len(payload.dive_ids),
        )
        return imported

    async def _render_dive(
        self, dive_id: int, sample_limit: Optional[int]
    ) -> List[CheckerboardLatticeRender]:
        """Stage one dive's frames, render its lattices, drop the scratch.

        The same command sequence as the calibration parent, because it is the
        same physical work: the board is only visible in pixels, so the frames
        have to reach the data-worker.
        """
        # The calibration resolver, reused rather than reimplemented. It already
        # selects exactly the frames the fit consumed — canonical images with a
        # live laser dot — and that population *is* the study's subject. A
        # second resolver would be free to drift from it, and a study of a
        # different set of frames than the fit saw answers nothing.
        inputs = await _dispatch.resolve_inputs(
            "resolve_checkerboard_calibration_inputs_activity",
            dive_id,
            PerformCheckerboardCalibrationInput,
        )
        if not inputs.images:
            workflow.logger.warning(
                "lattice verification resolved no frames; not staging dive_id=%d",
                dive_id,
            )
            return []

        child_payload = VerifyCheckerboardLatticeInput(
            dive_id=inputs.dive_id,
            camera_matrix=inputs.camera_matrix,
            distortion_coefficients=inputs.distortion_coefficients,
            target_rows=inputs.target_rows,
            target_cols=inputs.target_cols,
            square_size_m=inputs.square_size_m,
            images=inputs.images,
            sample_limit=sample_limit,
        )

        # Woken twice for the reason the calibration parent documents: staging
        # runs for many minutes on the api-worker's queue, during which the
        # data-processing queue looks idle to the +55 sweeper, which would
        # scale the CPU worker to zero and leave the child hanging on an
        # unserved queue.
        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        await _dispatch.wake_data_worker()

        owns_scratch = True
        try:
            dispatched = await _dispatch.dispatch_child(
                "VerifyCheckerboardLatticeWorkflow",
                child_payload,
                child_id=f"verify-checkerboard-lattice-{dive_id}",
                execution_timeout=timedelta(hours=2),
                result_type=List[CheckerboardLatticeRender],
            )
            if dispatched is _dispatch.CHILD_ALREADY_RUNNING:
                # Another run owns this dive's scratch. Leaving it alone is the
                # dive-442 lesson: deleting it pulls the `.ORF`s out from under
                # a child still reading them.
                owns_scratch = False
                workflow.logger.info(
                    "dive_id=%d already has a lattice child running; skipping",
                    dive_id,
                )
                return []
            return list(dispatched or [])
        finally:
            if owns_scratch:
                await _dispatch.cleanup_raw(dive_id)

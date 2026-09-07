"""Checkerboard laser-calibration parent workflow (api-worker side).

Picks the next HIGH-priority dive linked to a planar `CalibrationTarget`,
stages its raw `.ORF` bytes, and dispatches
`PerformCheckerboardCalibrationWorkflow` to the data-worker's CPU queue. The
result is an ordinary `LaserExtrinsics` row — stage 14, laser depth, the
`calibrated` flag and `calibration_dive_id` borrowing all consume it without
knowing a board was involved.

**Heavier than stage 13's parent, and the same shape as a preprocess parent.**
Stage 13 is selector -> child: pure math over already-stored slate labels, no
NAS, no object store. This one has to put the frames where the data-worker can
read them, because the board is only visible in pixels — so it is selector ->
resolver -> wake -> stage_raw -> child -> cleanup_raw, and it runs against the
CPU queue rather than the light one.

That staging is the expensive part and worth knowing about: the NAS reads at
about 1 MB/s and a frame is ~13 MB, so a 133-frame calibration folder is close
to half an hour of staging. It is bounded, though — the cohort excludes any
dive that already has extrinsics, so a dive passes through here exactly once,
and the corpus this serves is eleven dives.

Cluster-correctness invariants are the standing ones (CLAUDE.md): SKIP overlap
on the schedule so two selectors cannot race the same dive, a deterministic
child id, and `ALLOW_DUPLICATE` so a dive whose bad calibration was remediated
can be refitted. ALLOW_DUPLICATE is safe here for the same reasons it is in
stage 13: the cohort offers only dives with no extrinsics row,
`put_laser_extrinsics` upserts on dive_id, and the fit refuses to persist a
result that disagrees with its own dots.
"""

from datetime import timedelta

from fishsense_shared import PerformCheckerboardCalibrationInput
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class PerformCheckerboardCalibrationParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive needing checkerboard calibration
    and dispatch its fit to the data-worker.

    Returns the dive_id processed (or None when the backlog is empty). Each
    invocation drains exactly one dive.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_checkerboard_calibration_activity"
        )
        if dive_id is None:
            return None

        inputs = await _dispatch.resolve_inputs(
            "resolve_checkerboard_calibration_inputs_activity",
            dive_id,
            PerformCheckerboardCalibrationInput,
        )

        workflow.logger.info(
            "dispatching checkerboard calibration to data-worker "
            "dive_id=%d frames=%d board=%dx%d",
            inputs.dive_id,
            len(inputs.images),
            inputs.target_rows,
            inputs.target_cols,
        )

        if not inputs.images:
            # The cohort counted at least two dotted frames, so resolving none
            # means the resolver and the selector disagree. Staging the dive's
            # raw bytes from the NAS to dispatch nothing is the expensive way
            # to find that out, so stop here and let the mismatch show up as a
            # dive that keeps being selected and keeps doing nothing cheaply.
            workflow.logger.warning(
                "checkerboard calibration resolved no frames; "
                "not staging dive_id=%d",
                dive_id,
            )
            return inputs.dive_id

        # Woken twice, on purpose. The first call overlaps the pod's cold start
        # with staging, as every preprocess parent does.
        await _dispatch.wake_data_worker()
        await _dispatch.stage_raw(dive_id)
        # The second call is the one this stage needs. Staging here runs for
        # ~30 minutes on a 133-frame folder and happens on the *api-worker's*
        # queue, so `fishsense_data_processing_queue` has nothing Running the
        # whole time — and the +55 scale-to-zero sweeper reads exactly that as
        # idle. It would scale the CPU worker to 0 mid-staging and the child
        # would then land on an unserved queue: not a failure, a hang until the
        # 2h execution timeout, with SKIP overlap suppressing the firings
        # behind it. Likeliest precisely when these dives get their turn, since
        # the sweeper only scales down once the preprocess backlog has drained.
        #
        # The wake is an idempotent absolute target, so the extra call is a
        # no-op when the worker is already up and closes the window however
        # long staging took.
        await _dispatch.wake_data_worker()
        try:
            await _dispatch.dispatch_child(
                "PerformCheckerboardCalibrationWorkflow",
                inputs,
                child_id=f"perform-checkerboard-calibration-{dive_id}",
                # Generous: the fan-out is one rawpy decode plus a corner
                # search per frame, two at a time, over as many as 133 frames.
                execution_timeout=timedelta(hours=2),
            )
        finally:
            # Cleanup on failure too, which the preprocess parents do not do.
            # For them a child failure is unusual; here it is the *expected*
            # shape of "this dive cannot be calibrated from a board" — the fit
            # raises below `MIN_LASER_POINTS` — and the dive then stays in the
            # cohort and is re-selected hourly until an operator intervenes.
            # Leaving a dive's worth of `.ORF` scratch in Garage on each of
            # those firings buys nothing: it is reproducible from the NAS, and
            # the next firing re-stages it (cheaply, via the HEAD check) anyway.
            await _dispatch.cleanup_raw(dive_id)
        return inputs.dive_id

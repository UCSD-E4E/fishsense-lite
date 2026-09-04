"""Drain the auto-accept gate's backlog, one dive per firing.

The gate normally runs off the back of `PredictLaserImagesParentWorkflow`,
after it persists a dive's predictions — but only when the predict child
returned *new* ones. A dive that was already fully predicted never re-enters
the predict cohort, so it never produces results, so its predictions were never
judged. When the gate shipped in v2.19.0 that left **3,711 predictions across
~65 dives** permanently at `gate_verdict IS NULL`, and auto-accept reached
almost none of the backlog it was built for.

This is that backlog's drain. It is a cohort parent rather than a hand-run
backfill for the reason CLAUDE.md gives about selecting on mismatch: it empties
itself, stays empty afterwards, and re-arms on its own if anything ever leaves
a verdict NULL again — which a re-prediction does by design, since it clears
the verdict computed from a dot the row no longer holds.

Structurally the lightest parent in the tree, like stages 13 and 14: two
activity calls and a child. No NAS staging, no object store, no fan-out — the
data-worker child fetches the dive's predictions itself and the work is a line
fit.
"""

from fishsense_shared import GATE_CHILD_EXECUTION_TIMEOUT, LaserAutoAcceptSummary
from temporalio import workflow

from fishsense_api_workflow_worker.workflows import _dispatch


@workflow.defn
class EvaluateLaserAutoAcceptParentWorkflow:
    # pylint: disable=too-few-public-methods
    """Auto-pick the next HIGH-priority dive with unjudged laser predictions
    and run the gate over it. Returns the dive_id processed (or None when the
    backlog is empty) — each invocation drains exactly one dive.
    """

    @workflow.run
    async def run(self) -> int | None:
        dive_id = await _dispatch.select_dive(
            "select_next_high_priority_dive_for_laser_auto_accept_activity"
        )
        if dive_id is None:
            return None

        await _dispatch.wake_data_worker()
        summary: LaserAutoAcceptSummary = await _dispatch.dispatch_child(
            "EvaluateLaserAutoAcceptWorkflow",
            dive_id,
            child_id=f"auto-accept-laser-{dive_id}",
            # Shared with the predict parent, which dispatches the same child.
            # It must outlast the child's own activity budget so the activity's
            # timeout is the one that fires and names the bound that was hit.
            execution_timeout=GATE_CHILD_EXECUTION_TIMEOUT,
            result_type=LaserAutoAcceptSummary,
        )
        workflow.logger.info(
            "auto-accept backlog dive_id=%d enabled=%s eligible=%s reason=%s "
            "auto_accepted=%d/%d verdicts=%s",
            dive_id,
            summary.enabled,
            summary.eligible,
            summary.reason,
            summary.auto_accepted,
            sum(summary.verdicts.values()),
            summary.verdicts,
        )

        if summary.auto_accepted:
            # Judging alone changes the database and nothing a labeler sees:
            # populate imports a task once, and every dive in this backlog is
            # one whose tasks already exist. Attaching to them is what turns a
            # verdict into saved review time.
            applied = await _dispatch.run_sdk_activity(
                "apply_laser_auto_accept_for_dive_activity", dive_id
            )
            workflow.logger.info(
                "auto-accept backlog applied to existing tasks dive_id=%d applied=%s",
                dive_id,
                applied,
            )

        return dive_id

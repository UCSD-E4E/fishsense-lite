"""Workflow that decides which of a dive's laser predictions may skip review.

Thin wrapper around `evaluate_laser_auto_accept_activity`. Dispatched as a
child workflow by `PredictLaserImagesParentWorkflow` (api-worker) once the
dive's predictions have been persisted.

Same cross-worker split as the validation workflow: the api-worker owns the
schedule and knows which dive was just predicted, the data-worker owns the
math. The gate is numpy over the dive's own dots, so it belongs beside the
RANSAC kernel it shares with laser-label validation rather than on the
orchestration side.

Runs on the CPU queue, not the GPU one that produced the predictions. It is a
line fit; holding a contended NRP card through it would be waste, and the CPU
Deployment is the one every other math stage already uses.
"""

from datetime import timedelta

from fishsense_shared import LaserAutoAcceptSummary
from temporalio import workflow


@workflow.defn
class EvaluateLaserAutoAcceptWorkflow:
    # pylint: disable=too-few-public-methods
    """On-demand wrapper around `evaluate_laser_auto_accept_activity`.

    Returns the per-dive summary so the parent can log the verdict mix, which
    is the monitoring signal for this stage.
    """

    @workflow.run
    async def run(self, dive_id: int) -> LaserAutoAcceptSummary:
        # Same three-axis shape as the validation workflow, and for the same
        # reason: the dominant cost is `get_laser_predictions` over Traefik on
        # a large dive, not the fit. `heartbeat_timeout` turns a silent hang
        # in that fetch into a diagnosable timeout rather than a 10-minute
        # wait, and the activity pumps heartbeats across both the fetch and
        # the verdict writes.
        return await workflow.execute_activity(
            "evaluate_laser_auto_accept_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=1),
        )

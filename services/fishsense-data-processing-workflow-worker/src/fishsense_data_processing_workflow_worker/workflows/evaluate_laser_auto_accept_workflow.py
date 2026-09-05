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

from fishsense_shared import (
    GATE_ACTIVITY_TIMEOUT,
    GATE_EXECUTION_TIMEOUT,
    GATE_QUEUE_WAIT_TIMEOUT,
    LaserAutoAcceptSummary,
)
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
        # Queue wait and execution are bounded SEPARATELY, and that split is
        # the whole point. This inherited the validation workflow's single
        # `schedule_to_close_timeout=15m`, which conflates the two -- so on a
        # busy CPU queue the budget was spent waiting rather than working, and
        # the activity expired without ever running. Two of the backlog
        # drain's first three firings died that way on 2026-09-04, behind
        # multi-GB rawpy decodes holding all four slots on
        # `fishsense_data_processing_queue`, for a fit that takes under a
        # second.
        #
        # `schedule_to_close` is the sum, never less: an attempt that finally
        # gets a slot near the end of its patience still gets its full run
        # rather than being cut off mid-fit.
        #
        # The values are shared with the api-worker parents rather than spelled
        # here, because their child `execution_timeout` has to outlast this --
        # see `fishsense_shared.auto_accept_timeouts`.
        #
        # `heartbeat_timeout` is unchanged and is what keeps the tight
        # execution bound honest: the dominant cost once running is
        # `get_laser_predictions` over Traefik on a large dive, and the
        # activity pumps heartbeats across both the fetch and the verdict
        # writes, so a silent hang surfaces in a minute instead of ten.
        return await workflow.execute_activity(
            "evaluate_laser_auto_accept_activity",
            args=(dive_id,),
            schedule_to_start_timeout=GATE_QUEUE_WAIT_TIMEOUT,
            start_to_close_timeout=GATE_EXECUTION_TIMEOUT,
            schedule_to_close_timeout=GATE_ACTIVITY_TIMEOUT,
            heartbeat_timeout=timedelta(minutes=1),
        )

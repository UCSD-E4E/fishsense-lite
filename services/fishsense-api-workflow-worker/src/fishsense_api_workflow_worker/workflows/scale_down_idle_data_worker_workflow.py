"""Hourly sweeper that scales the NRP data-worker back to zero when idle.

Thin wrapper around ``scale_down_data_worker_if_idle_activity`` — the
activity does the work (query the data-worker task queue, scale the
Deployment to 0 if it's been quiet past the cooldown). Scheduled near
the end of the hour, after the last preprocess/calibration parent
firing, so it doesn't race a parent that's still scaling the
data-worker *up*.

A no-op (returns ``False``) when k8s scaling isn't configured.
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.workflows._retry_policies import (
        SCALING_RETRY_POLICY,
    )


@workflow.defn
class ScaleDownIdleDataWorkerWorkflow:
    # pylint: disable=too-few-public-methods
    """Scale the data-worker Deployment to 0 if its task queue is quiet.

    Returns ``True`` if it scaled down this run, ``False`` otherwise.
    """

    @workflow.run
    async def run(self) -> bool:
        return await workflow.execute_activity(
            "scale_down_data_worker_if_idle_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=5),
            retry_policy=SCALING_RETRY_POLICY,
        )

"""Workflow to record the distance to each laser dot in a dive."""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_data_processing_workflow_worker.activities.compute_laser_depths_activity import (  # noqa: E501  pylint: disable=line-too-long
        ComputeLaserDepthsResult,
    )


@workflow.defn
class ComputeLaserDepthsWorkflow:
    # pylint: disable=too-few-public-methods
    """Store a `LaserDepth` for every validated laser label on `dive_id`.

    Thin wrapper around `compute_laser_depths_activity`, in the shape of
    stages 13 and 14: no image bytes, no fan-out, just SDK math the
    data-worker owns because the projection kernel is in fishsense-core.

    Returns the per-dive summary so a caller can see how much was computed
    versus already current — a run that is entirely `skipped_current` means
    the dive is settled and nothing upstream has moved.

    Run after stage 13 (or after linking a calibration source): the activity
    raises when the dive has no resolvable `laser_extrinsics`.
    """

    @workflow.run
    async def run(self, dive_id: int) -> ComputeLaserDepthsResult:
        return await workflow.execute_activity(
            "compute_laser_depths_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=2),
        )

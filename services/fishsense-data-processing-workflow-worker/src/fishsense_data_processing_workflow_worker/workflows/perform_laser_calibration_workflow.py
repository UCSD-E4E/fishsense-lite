"""Workflow to compute laser extrinsics for a single dive (stage 13)."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class PerformLaserCalibrationWorkflow:
    # pylint: disable=too-few-public-methods
    """Compute laser extrinsics for `dive_id` from its slate-laser labels.

    On-demand wrapper around `perform_laser_calibration_activity`. Returns
    the persisted `LaserExtrinsics` row id, or None when the dive has no
    slate / no slate labels (genuine no-op).
    """

    @workflow.run
    async def run(self, dive_id: int) -> int | None:
        return await workflow.execute_activity(
            "perform_laser_calibration_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

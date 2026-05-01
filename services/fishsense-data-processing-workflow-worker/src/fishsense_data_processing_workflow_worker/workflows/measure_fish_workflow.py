"""Workflow to measure fish lengths for a dive (stage 14)."""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_data_processing_workflow_worker.activities.measure_fish_activity import (  # noqa: E501  pylint: disable=line-too-long
        MeasureFishResult,
    )


@workflow.defn
class MeasureFishWorkflow:
    # pylint: disable=too-few-public-methods
    """Compute Measurements for every top-three species label on `dive_id`.

    On-demand wrapper around `measure_fish_activity`. Returns the per-dive
    summary so callers can see how many measurements were written, how
    many got dropped to NaN, and how many were skipped for missing
    upstream context.

    Run after stages 12 (slate sync), 13 (laser calibration), and 6.1
    (LABEL_STUDIO clusters). Stage 13 is enforced by the activity (it
    raises if `laser_extrinsics` is missing); the others surface as
    counts in the result rather than hard errors.
    """

    @workflow.run
    async def run(self, dive_id: int) -> MeasureFishResult:
        return await workflow.execute_activity(
            "measure_fish_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(hours=1),
            heartbeat_timeout=timedelta(minutes=2),
        )

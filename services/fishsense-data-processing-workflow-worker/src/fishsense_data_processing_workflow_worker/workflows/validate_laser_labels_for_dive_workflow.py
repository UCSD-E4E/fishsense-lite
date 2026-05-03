"""Workflow that runs RANSAC line-fit validation for one dive's laser labels.

Thin wrapper around `validate_laser_labels_for_dive_activity`. Dispatched
as a child workflow by `SyncLabelStudioLaserLabelsWorkflow` (api-worker)
once a dive's laser labeling has fully completed.

Per the cross-worker convention in CLAUDE.md, the api-worker decides
*which* dives to validate (it owns the schedule + has the complete-dive
SDK call) and the data-worker does the math (it has numpy + the future
image-based-validation deps already wired in). Phase 1 is observe-only;
no `superseded` writes happen yet.
"""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class ValidateLaserLabelsForDiveWorkflow:
    # pylint: disable=too-few-public-methods
    """On-demand wrapper around `validate_laser_labels_for_dive_activity`.

    Returns the number of labels flagged as outliers (0 when the line
    isn't confident, no positives, etc.). Phase 1 logs flags only.
    """

    @workflow.run
    async def run(self, dive_id: int) -> int:
        return await workflow.execute_activity(
            "validate_laser_labels_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=5),
        )

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
        # Three-axis timeout shape:
        #   * `start_to_close` caps a single attempt — `get_laser_labels`
        #     on a large dive can return tens of MB of `label_studio_json`
        #     through Traefik, so 10m gives the SDK 10s-timeout × 3-retry
        #     ladder room to ride out a slow link without bailing early.
        #   * `heartbeat_timeout` (1m) fires if the activity stops
        #     heartbeating mid-fetch — converts a silent hang into a
        #     diagnosable timeout faster than waiting for start_to_close.
        #   * `schedule_to_close` (15m) bounds the whole thing including
        #     any retry the workflow might trigger.
        return await workflow.execute_activity(
            "validate_laser_labels_for_dive_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            start_to_close_timeout=timedelta(minutes=10),
            heartbeat_timeout=timedelta(minutes=1),
        )

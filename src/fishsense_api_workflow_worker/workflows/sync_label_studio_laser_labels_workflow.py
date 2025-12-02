"""Workflow to sync laser labels from Label Studio."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class SyncLabelStudioLaserLabelsWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to sync laser labels from Label Studio."""

    @workflow.run
    async def run(self):
        """Run the workflow to sync laser labels from Label Studio."""
        await workflow.execute_activity(
            "sync_users_label_studio_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

"""Workflow to update dashboard configuration."""

from datetime import timedelta

from temporalio import workflow


@workflow.defn
class UpdateDashboardConfigWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow to update dashboard configuration."""

    @workflow.run
    async def run(self):
        """Run the workflow to update dashboard configuration."""
        (
            laser_label_projects,
            species_label_projects,
            head_tail_label_projects,
            slate_label_projects,
        ) = await workflow.execute_activity(
            "get_label_studio_projects_activity",
            args=(),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

        await workflow.execute_activity(
            "write_dashboard_config_activity",
            args=(
                laser_label_projects,
                species_label_projects,
                head_tail_label_projects,
                slate_label_projects,
            ),
            schedule_to_close_timeout=timedelta(minutes=10),
        )

"""Workflow to reconcile species labels into LABEL_STUDIO clusters for one dive.

On-demand (no schedule) — operator triggers per dive once species
labeling is complete. Stage 14 measurement (`measure_fish_activity`)
depends on the LABEL_STUDIO clusters this workflow creates.
"""

from datetime import timedelta

from temporalio import workflow

# The activity module transitively imports the fishsense-api-sdk
# `Client`, which pulls in `urllib.request` — Temporal's workflow
# sandbox blocks that import on workflow validation. Pass-through tells
# the sandbox to import it as-is; the workflow only uses the dataclass
# as a return-type annotation, so no nondeterministic code runs.
with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.activities.update_dive_image_groups_activity import (  # noqa: E501  pylint: disable=line-too-long
        UpdateDiveImageGroupsResult,
    )


@workflow.defn
class UpdateDiveImageGroupsWorkflow:
    # pylint: disable=too-few-public-methods
    """Workflow port of stage 6.1 update_dive_image_groups."""

    @workflow.run
    async def run(self, dive_id: int) -> UpdateDiveImageGroupsResult:
        """Reconcile `dive_id`'s species labels into LABEL_STUDIO clusters."""
        return await workflow.execute_activity(
            "update_dive_image_groups_activity",
            args=(dive_id,),
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=2),
        )

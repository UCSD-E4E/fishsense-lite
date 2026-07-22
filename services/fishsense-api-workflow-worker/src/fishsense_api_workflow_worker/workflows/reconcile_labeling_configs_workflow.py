"""Hourly pass that converges per-dive LS projects onto the current configs.

Thin wrapper around ``reconcile_labeling_configs_activity`` — the activity
does the work (enumerate the workspace, match each project's title suffix to
the stage that owns it, push the config when it has drifted).

This exists because the heal built into
``create_or_get_label_studio_project`` only runs during populate, and
populate stops dispatching for a dive once it is fully populated. Finished
projects — the stable ones labelers actually work in — therefore never saw a
taxonomy change. Walking the projects on a schedule reaches them regardless
of any cohort.

Idempotent: a pass with no drift updates nothing, so re-running is free.
"""

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from fishsense_api_workflow_worker.activities.reconcile_labeling_configs_activity import (  # pylint: disable=line-too-long
        ReconcileLabelingConfigsResult,
    )


@workflow.defn
class ReconcileLabelingConfigsWorkflow:
    # pylint: disable=too-few-public-methods
    """Push the current labeling config onto every per-dive project."""

    @workflow.run
    async def run(self) -> ReconcileLabelingConfigsResult:
        return await workflow.execute_activity(
            "reconcile_labeling_configs_activity",
            args=(),
            # Sized for a workspace-wide walk: one list call plus a detail
            # fetch per project whose list entry omits `label_config`. The
            # activity heartbeats per project, so a slow LS shows up as a
            # heartbeat timeout rather than silently eating the whole window.
            schedule_to_close_timeout=timedelta(minutes=15),
            heartbeat_timeout=timedelta(minutes=2),
        )

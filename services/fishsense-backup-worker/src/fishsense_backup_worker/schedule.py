"""Build the Temporal Schedule registered by the backup worker.

The idempotent registration call (`ensure_schedule`) lives in
`fishsense_shared.temporal`. This module just wires the backup-specific
inputs into a `Schedule` value the worker passes to it.
"""

from typing import List

from temporalio.client import (
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleSpec,
)

from fishsense_backup_worker.workflows.backup_databases_workflow import (
    BackupDatabasesInput,
    BackupDatabasesWorkflow,
)


def build_backup_schedule(
    *,
    databases: List[str],
    nas_root_path: str,
    retention_count: int,
    cron_expression: str,
    task_queue: str,
    workflow_id: str,
) -> Schedule:
    """Construct the Schedule object the backup worker registers on
    startup. Broken out from registration so it's unit-testable without
    a Temporal cluster."""
    return Schedule(
        action=ScheduleActionStartWorkflow(
            BackupDatabasesWorkflow.run,
            BackupDatabasesInput(
                databases=databases,
                nas_root_path=nas_root_path,
                retention_count=retention_count,
            ),
            id=workflow_id,
            task_queue=task_queue,
        ),
        spec=ScheduleSpec(cron_expressions=[cron_expression]),
    )

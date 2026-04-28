"""Idempotent Temporal Schedule registration for the backup workflow.

Worker startup calls `ensure_backup_schedule` — if the schedule
already exists (e.g. previous deploy created it), we leave it alone.
Updates require an explicit `temporal schedule delete` and a
re-deploy. Keeping the rule simple avoids surprising ops behavior
when config like the cron string changes.
"""

import logging
from typing import List

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleAlreadyRunningError,
    ScheduleSpec,
)

from fishsense_backup_worker.workflows.backup_databases_workflow import (
    BackupDatabasesInput,
    BackupDatabasesWorkflow,
)

_log = logging.getLogger(__name__)


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
    startup. Broken out from the registration so it's unit-testable
    without a Temporal cluster."""
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


async def ensure_backup_schedule(
    client: Client,
    *,
    schedule_id: str,
    schedule: Schedule,
) -> None:
    """Create the schedule if it doesn't exist; otherwise no-op.

    We deliberately do NOT update existing schedules — operators have
    to delete + redeploy if they want config changes (cron, retention,
    db list) to take effect, so a config typo can't silently retire
    an existing schedule.
    """
    try:
        await client.create_schedule(schedule_id, schedule)
        _log.info("created backup schedule %s", schedule_id)
    except ScheduleAlreadyRunningError:
        _log.info(
            "backup schedule %s already exists; leaving as-is "
            "(delete + redeploy to pick up config changes)",
            schedule_id,
        )

"""Tests for the backup-worker's `build_backup_schedule`.

The idempotent registration helper (`ensure_schedule`) is owned by
`fishsense_shared`; its tests live alongside it. This file only covers
the backup-specific Schedule construction.
"""

from temporalio.client import Schedule, ScheduleActionStartWorkflow

from fishsense_backup_worker.schedule import build_backup_schedule


def test_build_backup_schedule_wires_inputs_into_action():
    schedule = build_backup_schedule(
        databases=["fishsense", "superset"],
        nas_root_path="/fishsense_backups",
        retention_count=14,
        cron_expression="0 3 * * *",
        task_queue="fishsense_backup_queue",
        workflow_id="fishsense-daily-db-backup",
    )

    assert isinstance(schedule, Schedule)
    action = schedule.action
    assert isinstance(action, ScheduleActionStartWorkflow)
    # pylint: disable=no-member
    # The isinstance check above narrows `action` to ScheduleActionStartWorkflow,
    # which has `.id` and `.task_queue`. Pylint can't narrow through the assert.
    assert action.id == "fishsense-daily-db-backup"
    assert action.task_queue == "fishsense_backup_queue"
    assert schedule.spec.cron_expressions == ["0 3 * * *"]

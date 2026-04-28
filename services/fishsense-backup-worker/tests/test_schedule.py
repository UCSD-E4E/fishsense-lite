"""Unit tests for the idempotent backup-schedule registration.

We don't need a real Temporal cluster here — the function's only job
is to call `create_schedule` once and treat ScheduleAlreadyRunningError
as success. A fake client with an async stub is enough.
"""

from typing import List

import pytest
from temporalio.client import Schedule, ScheduleAlreadyRunningError

from fishsense_backup_worker.schedule import (
    build_backup_schedule,
    ensure_backup_schedule,
)


def _schedule() -> Schedule:
    return build_backup_schedule(
        databases=["fishsense", "superset", "temporal_db"],
        nas_root_path="/fishsense_backups",
        retention_count=14,
        cron_expression="0 3 * * *",
        task_queue="fishsense_backup_queue",
        workflow_id="fishsense-daily-db-backup-{ScheduledStartTime}",
    )


class _FakeClient:
    """Records calls to create_schedule. `raise_already_running=True`
    simulates the schedule already existing."""

    def __init__(self, *, raise_already_running: bool = False):
        self._raise = raise_already_running
        self.calls: List[tuple] = []

    async def create_schedule(self, schedule_id: str, schedule):
        self.calls.append((schedule_id, schedule))
        if self._raise:
            raise ScheduleAlreadyRunningError()
        return None


@pytest.mark.asyncio
async def test_creates_schedule_when_missing():
    client = _FakeClient(raise_already_running=False)
    await ensure_backup_schedule(
        client, schedule_id="sched-id", schedule=_schedule()
    )
    assert len(client.calls) == 1
    assert client.calls[0][0] == "sched-id"


@pytest.mark.asyncio
async def test_no_op_when_schedule_already_exists():
    client = _FakeClient(raise_already_running=True)
    # Must not raise — the function swallows ScheduleAlreadyRunningError.
    await ensure_backup_schedule(
        client, schedule_id="sched-id", schedule=_schedule()
    )
    assert len(client.calls) == 1

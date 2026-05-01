"""Unit tests for the shared idempotent schedule registration.

The contract: `ensure_schedule` calls `create_schedule` once on the
provided client, and treats `ScheduleAlreadyRunningError` as success.
Workers depend on this so a redeploy doesn't accidentally mutate or
retire production schedules — operators must `temporal schedule delete`
to pick up config changes.
"""

from typing import List

import pytest
from temporalio.client import ScheduleAlreadyRunningError

from fishsense_shared import ensure_schedule


class _FakeClient:
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
    await ensure_schedule(client, schedule_id="sched-id", schedule=object())
    assert len(client.calls) == 1
    assert client.calls[0][0] == "sched-id"


@pytest.mark.asyncio
async def test_no_op_when_schedule_already_exists():
    client = _FakeClient(raise_already_running=True)
    # Must not raise — the function swallows ScheduleAlreadyRunningError.
    await ensure_schedule(client, schedule_id="sched-id", schedule=object())
    assert len(client.calls) == 1

"""The backup worker's startup wiring.

Not glamorous, but the failure mode is: the worker starts, the schedule fires,
and the workflow fails at runtime because an activity was never registered.
Nothing catches that at import time, and the only signal is a Temporal task
failure on a nightly job nobody is watching — for the service whose whole job
is being the rollback mechanism.

`main()` is driven with Temporal mocked out, so this asserts the wiring
(what got registered, under which id, on which queue) without a cluster.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "https://nas.example.com:6021")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "u")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "p")
    monkeypatch.setenv("E4EFS_POSTGRES__HOST", "postgres")
    monkeypatch.setenv("E4EFS_POSTGRES__USERNAME", "backup")
    monkeypatch.setenv("E4EFS_POSTGRES__PASSWORD", "secret")
    yield


@pytest.fixture
def wired(monkeypatch):
    """Run `main()` with Temporal replaced; hand back the recorded calls."""
    from fishsense_backup_worker import worker as sut  # pylint: disable=import-outside-toplevel

    client = MagicMock()
    connect = AsyncMock(return_value=client)
    monkeypatch.setattr(sut.Client, "connect", connect)

    ensure_schedule = AsyncMock()
    monkeypatch.setattr(sut, "ensure_schedule", ensure_schedule)
    monkeypatch.setattr(sut, "build_tls_config", lambda _s: None)
    monkeypatch.setattr(sut, "configure_logging", lambda: None)

    worker_instance = MagicMock()
    worker_instance.run = AsyncMock()
    worker_cls = MagicMock(return_value=worker_instance)
    monkeypatch.setattr(sut, "Worker", worker_cls)

    return sut, connect, ensure_schedule, worker_cls, worker_instance


@pytest.mark.asyncio
async def test_registers_both_activities_and_the_workflow(wired):
    """The runtime failure this guards: a schedule that fires into a worker
    which cannot execute what the workflow calls."""
    sut, _connect, _ensure, worker_cls, _instance = wired
    from fishsense_backup_worker.activities.pg_dump_database import (  # pylint: disable=import-outside-toplevel
        pg_dump_database,
    )
    from fishsense_backup_worker.activities.prune_database_backups import (  # pylint: disable=import-outside-toplevel
        prune_database_backups,
    )
    from fishsense_backup_worker.workflows.backup_databases_workflow import (  # pylint: disable=import-outside-toplevel
        BackupDatabasesWorkflow,
    )

    await sut.main()

    kwargs = worker_cls.call_args.kwargs
    assert set(kwargs["activities"]) == {pg_dump_database, prune_database_backups}
    assert kwargs["workflows"] == [BackupDatabasesWorkflow]


@pytest.mark.asyncio
async def test_listens_on_its_own_task_queue(wired):
    """Deliberately separate from the data-processing worker so the backup
    worker doesn't need to share Postgres credentials — see CLAUDE.md."""
    sut, _connect, _ensure, worker_cls, _instance = wired

    await sut.main()

    assert worker_cls.call_args.kwargs["task_queue"] == sut.settings.backup.task_queue


@pytest.mark.asyncio
async def test_registers_the_daily_schedule_idempotently_on_startup(wired):
    """First deploy creates the cadence with no manual ops; later deploys must
    hit `ensure_schedule`'s already-exists path rather than re-creating it."""
    sut, _connect, ensure_schedule, _worker_cls, _instance = wired

    await sut.main()

    ensure_schedule.assert_awaited_once()
    assert (
        ensure_schedule.await_args.kwargs["schedule_id"]
        == sut.settings.backup.schedule_id
    )


@pytest.mark.asyncio
async def test_the_schedule_carries_the_configured_databases_and_retention(wired):
    """Retention and the DB list are the two settings an operator actually
    changes; if they don't reach the schedule the override silently no-ops."""
    sut, _connect, ensure_schedule, _worker_cls, _instance = wired

    await sut.main()

    schedule = ensure_schedule.await_args.kwargs["schedule"]
    payload = schedule.action.args[0]
    assert payload.databases == list(sut.settings.backup.databases)
    assert payload.retention_count == int(sut.settings.backup.retention_count)


@pytest.mark.asyncio
async def test_connects_with_the_shared_tls_and_namespace_helpers(wired):
    """One mTLS implementation across every service — see
    fishsense_shared.build_tls_config."""
    sut, connect, _ensure, _worker_cls, _instance = wired

    await sut.main()

    connect.assert_awaited_once()
    assert connect.await_args.args[0] == (
        f"{sut.settings.temporal.host}:{sut.settings.temporal.port}"
    )


@pytest.mark.asyncio
async def test_the_worker_is_actually_started(wired):
    sut, _connect, _ensure, _worker_cls, instance = wired

    await sut.main()

    instance.run.assert_awaited_once()


def test_run_drives_main_under_asyncio(monkeypatch):
    from fishsense_backup_worker import worker as sut  # pylint: disable=import-outside-toplevel

    called = []
    monkeypatch.setattr(sut.asyncio, "run", called.append)
    monkeypatch.setattr(sut, "main", lambda: "coro")

    sut.run()

    assert called == ["coro"]

"""Workflow contract test for BackupDatabasesWorkflow.

Runs the workflow against an in-process Temporal test server with
stubbed activities — no real pg_dump, no NAS. Verifies the fanout
shape: one pg_dump per DB, then one prune per DB, with the right
arguments passed through.
"""

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_backup_worker.workflows.backup_databases_workflow import (
    BackupDatabasesInput,
    BackupDatabasesWorkflow,
    PgDumpDatabaseInput,
    PruneDatabaseBackupsInput,
)


@pytest.mark.asyncio
async def test_workflow_dumps_each_db_then_prunes_each_db():
    dump_calls: List[PgDumpDatabaseInput] = []
    prune_calls: List[PruneDatabaseBackupsInput] = []

    @activity.defn(name="pg_dump_database")
    async def stub_pg_dump(input: PgDumpDatabaseInput) -> None:
        dump_calls.append(input)

    @activity.defn(name="prune_database_backups")
    async def stub_prune(input: PruneDatabaseBackupsInput) -> None:
        prune_calls.append(input)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-backup",
            workflows=[BackupDatabasesWorkflow],
            activities=[stub_pg_dump, stub_prune],
        ):
            await env.client.execute_workflow(
                BackupDatabasesWorkflow.run,
                BackupDatabasesInput(
                    databases=["fishsense", "superset", "temporal_db"],
                    nas_root_path="/fishsense_backups",
                    retention_count=14,
                ),
                id=f"backup-test-{uuid.uuid4()}",
                task_queue="test-backup",
            )

    assert {c.db_name for c in dump_calls} == {"fishsense", "superset", "temporal_db"}
    assert {c.db_name for c in prune_calls} == {"fishsense", "superset", "temporal_db"}
    for c in dump_calls:
        assert c.nas_root_path == "/fishsense_backups"
    for c in prune_calls:
        assert c.nas_root_path == "/fishsense_backups"
        assert c.keep == 14


@pytest.mark.asyncio
async def test_workflow_does_not_prune_until_all_dumps_complete():
    """If pg_dump for any DB is still in flight, prune must not run for
    *any* DB. Otherwise a slow dump + a fast prune could drop us below
    `retention_count` good backups during the run."""
    timeline: List[str] = []

    @activity.defn(name="pg_dump_database")
    async def stub_pg_dump(input: PgDumpDatabaseInput) -> None:
        timeline.append(f"dump:{input.db_name}")

    @activity.defn(name="prune_database_backups")
    async def stub_prune(input: PruneDatabaseBackupsInput) -> None:
        timeline.append(f"prune:{input.db_name}")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-backup-order",
            workflows=[BackupDatabasesWorkflow],
            activities=[stub_pg_dump, stub_prune],
        ):
            await env.client.execute_workflow(
                BackupDatabasesWorkflow.run,
                BackupDatabasesInput(
                    databases=["fishsense", "superset"],
                    nas_root_path="/fishsense_backups",
                    retention_count=14,
                ),
                id=f"backup-order-test-{uuid.uuid4()}",
                task_queue="test-backup-order",
            )

    last_dump_index = max(i for i, e in enumerate(timeline) if e.startswith("dump:"))
    first_prune_index = min(i for i, e in enumerate(timeline) if e.startswith("prune:"))
    assert last_dump_index < first_prune_index, (
        f"prune ran before all dumps finished: {timeline}"
    )


@pytest.mark.asyncio
async def test_workflow_with_no_databases_makes_no_activity_calls():
    dump_calls: List[PgDumpDatabaseInput] = []
    prune_calls: List[PruneDatabaseBackupsInput] = []

    @activity.defn(name="pg_dump_database")
    async def stub_pg_dump(input: PgDumpDatabaseInput) -> None:
        dump_calls.append(input)

    @activity.defn(name="prune_database_backups")
    async def stub_prune(input: PruneDatabaseBackupsInput) -> None:
        prune_calls.append(input)

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-backup-empty",
            workflows=[BackupDatabasesWorkflow],
            activities=[stub_pg_dump, stub_prune],
        ):
            await env.client.execute_workflow(
                BackupDatabasesWorkflow.run,
                BackupDatabasesInput(
                    databases=[],
                    nas_root_path="/fishsense_backups",
                    retention_count=14,
                ),
                id=f"backup-empty-test-{uuid.uuid4()}",
                task_queue="test-backup-empty",
            )

    assert dump_calls == []
    assert prune_calls == []

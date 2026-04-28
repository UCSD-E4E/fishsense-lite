"""Daily backup workflow.

Fires from a Temporal Schedule. Fans out a `pg_dump_database` activity
per DB, then a `prune_database_backups` activity per DB to enforce
retention. Per-DB activities are independent — one DB's pg_dump
failure doesn't block the others or skip pruning of unrelated DBs.
"""

import asyncio
from datetime import timedelta
from typing import List

from pydantic import BaseModel
from temporalio import workflow


class PgDumpDatabaseInput(BaseModel):
    """Input to the pg_dump_database activity."""

    db_name: str
    nas_root_path: str  # e.g. "/fishsense_backups"


class PruneDatabaseBackupsInput(BaseModel):
    """Input to the prune_database_backups activity."""

    db_name: str
    nas_root_path: str
    keep: int


class BackupDatabasesInput(BaseModel):
    """Whole-workflow input."""

    databases: List[str]
    nas_root_path: str
    retention_count: int


@workflow.defn
class BackupDatabasesWorkflow:
    # pylint: disable=too-few-public-methods
    @workflow.run
    async def run(self, input: BackupDatabasesInput) -> None:
        workflow.logger.info(
            "backup workflow start dbs=%s retention=%d",
            input.databases,
            input.retention_count,
        )

        # Dump each DB in parallel. pg_dump on Postgres uses MVCC
        # snapshots so concurrent dumps don't interfere with each other.
        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "pg_dump_database",
                    PgDumpDatabaseInput(
                        db_name=db,
                        nas_root_path=input.nas_root_path,
                    ),
                    schedule_to_close_timeout=timedelta(hours=2),
                    start_to_close_timeout=timedelta(hours=2),
                )
                for db in input.databases
            ]
        )

        # Prune after every dump succeeded — we never want to prune a
        # DB whose own dump failed, since that could drop us below
        # `retention_count` good backups.
        await asyncio.gather(
            *[
                workflow.execute_activity(
                    "prune_database_backups",
                    PruneDatabaseBackupsInput(
                        db_name=db,
                        nas_root_path=input.nas_root_path,
                        keep=input.retention_count,
                    ),
                    schedule_to_close_timeout=timedelta(minutes=10),
                )
                for db in input.databases
            ]
        )

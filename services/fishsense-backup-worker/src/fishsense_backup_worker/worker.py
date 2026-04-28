"""Backup worker entrypoint.

Connects to the same Temporal cluster as the rest of the workers but
listens on its own task queue (`fishsense_backup_queue` by default).
On startup, idempotently registers the daily backup Schedule so the
first deploy of this service kicks off the cadence with no manual ops.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fishsense_shared import build_tls_config
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_backup_worker.activities.pg_dump_database import (
    pg_dump_database,
)
from fishsense_backup_worker.activities.prune_database_backups import (
    prune_database_backups,
)
from fishsense_backup_worker.config import configure_logging, settings
from fishsense_backup_worker.schedule import (
    build_backup_schedule,
    ensure_backup_schedule,
)
from fishsense_backup_worker.workflows.backup_databases_workflow import (
    BackupDatabasesWorkflow,
)


async def main() -> None:
    """Main entry point for the backup worker."""
    configure_logging()
    log = logging.getLogger()

    tls_config = build_tls_config(settings.temporal)

    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}", tls=tls_config
    )

    schedule = build_backup_schedule(
        databases=list(settings.backup.databases),
        nas_root_path=settings.backup.nas_root_path,
        retention_count=int(settings.backup.retention_count),
        cron_expression=settings.backup.schedule_cron,
        task_queue=settings.backup.task_queue,
        workflow_id="fishsense-daily-db-backup-{ScheduledStartTime}",
    )
    await ensure_backup_schedule(
        client, schedule_id=settings.backup.schedule_id, schedule=schedule
    )

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        worker = Worker(
            client,
            task_queue=settings.backup.task_queue,
            workflows=[BackupDatabasesWorkflow],
            activity_executor=executor,
            activities=[pg_dump_database, prune_database_backups],
        )

        log.info(
            "fishsense-backup-worker started: queue=%s schedule=%s dbs=%s",
            settings.backup.task_queue,
            settings.backup.schedule_id,
            list(settings.backup.databases),
        )
        await worker.run()


def run() -> None:
    """Run the worker."""
    asyncio.run(main())

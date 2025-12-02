"""Worker for FishSense API Workflow"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
    TLSConfig,
)
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.get_laser_label_studio_project_ids_activity import (
    get_laser_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.sync_laser_labels_for_label_studio_project_activity import (  # pylint: disable=line-too-long
    sync_laser_labels_for_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.sync_users_label_studio_activity import (
    sync_users_label_studio_activity,
)
from fishsense_api_workflow_worker.config import configure_logging, settings
from fishsense_api_workflow_worker.workflows.sync_label_studio_laser_labels_workflow import (
    SyncLabelStudioLaserLabelsWorkflow,
)

TASK_QUEUE_NAME = "fishsense_api_queue"


async def schedule_exists(client: Client, schedule_id: str) -> bool:
    """Check if a schedule exists."""
    schedules = await client.list_schedules()
    async for s in schedules:
        if s.id == schedule_id:
            return True

    # If we reach here, no schedule with the given ID was found
    logging.info("No schedule found with ID: %s", schedule_id)


async def schedule_sync_label_studio_laser_labels_workflow(client: Client):
    """Schedule the SyncLabelStudioLaserLabelsWorkflow to run periodically."""
    schedule_id = "sync-label-studio-laser-labels-workflow-schedule"

    if await schedule_exists(client, schedule_id):
        logging.info("Schedule %s already exists, skipping...", schedule_id)
        return

    await client.create_schedule(
        schedule_id,
        Schedule(
            action=ScheduleActionStartWorkflow(
                SyncLabelStudioLaserLabelsWorkflow.run,
                args=(),
                id="sync-label-studio-laser-labels-workflow",
                task_queue=TASK_QUEUE_NAME,
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(hours=1))]
            ),
            state=ScheduleState(),
        ),
    )


async def schedule_workflows(client: Client):
    """Schedule workflows for the worker."""

    async with asyncio.TaskGroup() as tg:
        tg.create_task(schedule_sync_label_studio_laser_labels_workflow(client))


async def main():
    """Main entry point for the worker."""

    configure_logging()
    log = logging.getLogger()

    tls_config: TLSConfig | None = None
    if settings.temporal.tls:
        with Path(settings.temporal.client_cert).open("rb") as f:
            client_cert = f.read()
        with Path(settings.temporal.client_private_key).open("rb") as f:
            client_private_key = f.read()

        server_root_ca_cert: bytes | None = None
        if "server_root_ca_cert" in settings.temporal:
            with Path(settings.temporal.server_root_ca_cert).open("rb") as f:
                server_root_ca_cert = f.read()

        tls_config = TLSConfig(
            client_cert=client_cert,
            client_private_key=client_private_key,
            server_root_ca_cert=server_root_ca_cert,
            domain=settings.temporal.domain if "domain" in settings.temporal else None,
        )

    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}", tls=tls_config
    )

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE_NAME,
            workflows=[
                SyncLabelStudioLaserLabelsWorkflow,
            ],
            activity_executor=executor,
            activities=[
                get_laser_label_studio_project_ids_activity,
                sync_laser_labels_for_label_studio_project_activity,
                sync_users_label_studio_activity,
            ],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

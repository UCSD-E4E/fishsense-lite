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

from fishsense_api_workflow_worker.activities.cluster_dive_frames import cluster_dive_frames
from fishsense_api_workflow_worker.activities.select_dives import select_dives
from fishsense_api_workflow_worker.activities.store_dive_clusters import store_dive_clusters
from fishsense_api_workflow_worker.activities.sync_label_studio_head_tail_labels import (
    sync_label_studio_head_tail_labels,
)
from fishsense_api_workflow_worker.activities.sync_label_studio_laser_labels import (
    sync_label_studio_laser_labels,
)
from fishsense_api_workflow_worker.activities.sync_users_into_postgres import (
    sync_users_into_postgres,
)
from fishsense_api_workflow_worker.config import (
    PG_CONNECTION_STRING,
    configure_logging,
    settings,
)
from fishsense_api_workflow_worker.database import Database
from fishsense_api_workflow_worker.workflows.ingest_dives import IngestDivesWorkflow
from fishsense_api_workflow_worker.workflows.read_label_studio_head_tail_labels import (
    ReadLabelStudioHeadTailLabelsWorkflow,
)
from fishsense_api_workflow_worker.workflows.read_label_studio_laser_labels import (
    ReadLabelStudioLaserLabelsWorkflow,
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


async def schedule_read_label_studio_laser_label_workflows(client: Client):
    """Schedule workflows to read laser labels from Label Studio."""
    for laser_project_id in settings.label_studio.laser_project_ids:
        schedule_id = f"read-label-studio-laser-labels-schedule-{laser_project_id}"

        if await schedule_exists(client, schedule_id):
            logging.info("Schedule %s already exists, skipping...", schedule_id)
            continue

        await client.create_schedule(
            schedule_id,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    ReadLabelStudioLaserLabelsWorkflow.run,
                    args=(
                        settings.label_studio.host,
                        settings.label_studio.api_key,
                        PG_CONNECTION_STRING,
                        laser_project_id,
                    ),
                    id=f"read-label-studio-laser-labels-workflow-{laser_project_id}",
                    task_queue=TASK_QUEUE_NAME,
                ),
                spec=ScheduleSpec(
                    intervals=[ScheduleIntervalSpec(every=timedelta(hours=1))]
                ),
                state=ScheduleState(),
            ),
        )


async def schedule_read_label_studio_head_tail_label_workflows(client: Client):
    """Schedule workflows to read head-tail labels from Label Studio."""
    for head_tail_project_id in settings.label_studio.head_tail_project_ids:
        schedule_id = (
            f"read-label-studio-head-tail-labels-schedule-{head_tail_project_id}"
        )

        if await schedule_exists(client, schedule_id):
            logging.info("Schedule %s already exists, skipping...", schedule_id)
            continue

        await client.create_schedule(
            schedule_id,
            Schedule(
                action=ScheduleActionStartWorkflow(
                    ReadLabelStudioHeadTailLabelsWorkflow.run,
                    args=(
                        settings.label_studio.host,
                        settings.label_studio.api_key,
                        PG_CONNECTION_STRING,
                        head_tail_project_id,
                    ),
                    id=f"read-label-studio-head-tail-labels-workflow-{head_tail_project_id}",
                    task_queue=TASK_QUEUE_NAME,
                ),
                spec=ScheduleSpec(
                    intervals=[ScheduleIntervalSpec(every=timedelta(hours=1))]
                ),
                state=ScheduleState(),
            ),
        )


async def schedule_ingest_dives_workflow(client: Client):
    schedule_id = f"ingest-dives-schedule"

    if await schedule_exists(client, schedule_id):
        logging.info("Schedule %s already exists, skipping...", schedule_id)
        return

    await client.create_schedule(
        schedule_id,
        Schedule(
            action=ScheduleActionStartWorkflow(
                IngestDivesWorkflow.run,
                args=(
                    PG_CONNECTION_STRING,
                    settings.temporal.host,
                    settings.temporal.port,
                    settings.temporal.tls,
                    (
                        settings.temporal.client_certschedule_dive_frame_grouping
                        if "client_cert" in settings.temporal
                        else None
                    ),
                    (
                        settings.temporal.client_private_key
                        if "client_private_key" in settings.temporal
                        else None
                    ),
                    (
                        settings.temporal.server_root_ca_cert
                        if "server_root_ca_cert" in settings.temporal
                        else None
                    ),
                    (
                        settings.temporal.domain
                        if "domain" in settings.temporschedule_dive_frame_groupingal
                        else None
                    ),
                ),
                id=f"ingest-dives-workflow",
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
        tg.create_task(schedule_ingest_dives_workflow(client))


# async def schedule_workflows(client: Client):
#     """Schedule workflows for the worker."""

#     async with asyncio.TaskGroup() as tg:
#         tg.create_task(schedule_read_label_studio_laser_label_workflows(client))
#         tg.create_task(schedule_read_label_studio_head_tail_label_workflows(client))


async def main():
    """Main entry point for the worker."""

    configure_logging()
    log = logging.getLogger()

    database = Database(PG_CONNECTION_STRING)
    await database.init_database()

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
                IngestDivesWorkflow,
                ReadLabelStudioLaserLabelsWorkflow,
                ReadLabelStudioHeadTailLabelsWorkflow,
            ],
            activity_executor=executor,
            activities=[
                select_dives,
                cluster_dive_frames,
                store_dive_clusters,

                sync_label_studio_laser_labels,
                sync_label_studio_head_tail_labels,
                sync_users_into_postgres,
            ],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

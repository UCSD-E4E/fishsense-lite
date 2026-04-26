"""Worker for FishSense API Workflow"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Callable

from fishsense_shared import build_tls_config
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
)
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.get_headtail_label_studio_project_ids_activity import (  # pylint: disable=line-too-long
    get_headtail_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_label_studio_projects_activity import (
    get_label_studio_projects_activity,
)
from fishsense_api_workflow_worker.activities.get_laser_label_studio_project_ids_activity import (
    get_laser_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.sync_headtail_labels_for_label_studio_project_activity import (  # pylint: disable=line-too-long
    sync_headtail_labels_for_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.sync_laser_labels_for_label_studio_project_activity import (  # pylint: disable=line-too-long
    sync_laser_labels_for_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.sync_users_label_studio_activity import (
    sync_users_label_studio_activity,
)
from fishsense_api_workflow_worker.activities.write_dashboard_config_activity import (
    write_dashboard_config_activity,
)
from fishsense_shared import ExceptionGroupErrorLogging
from fishsense_api_workflow_worker.config import configure_logging, settings
from fishsense_api_workflow_worker.workflows.sync_label_studio_headtail_labels_workflow import (
    SyncLabelStudioHeadTailLabelsWorkflow,
)
from fishsense_api_workflow_worker.workflows.sync_label_studio_laser_labels_workflow import (
    SyncLabelStudioLaserLabelsWorkflow,
)
from fishsense_api_workflow_worker.workflows.update_dashboard_config_workflow import (
    UpdateDashboardConfigWorkflow,
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


async def schedule_workflow(
    client: Client, schedule_id: str, workflow_cls: Callable, interval: timedelta
):
    """Schedule a workflow to run periodically."""
    if await schedule_exists(client, schedule_id):
        logging.info("Schedule %s already exists, skipping...", schedule_id)
        return

    await client.create_schedule(
        schedule_id,
        Schedule(
            action=ScheduleActionStartWorkflow(
                workflow_cls.run,
                args=(),
                id=f"{workflow_cls.__name__}-workflow",
                task_queue=TASK_QUEUE_NAME,
                run_timeout=timedelta(minutes=30),
            ),
            spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=interval)]),
            state=ScheduleState(),
        ),
    )


async def schedule_workflows(client: Client):
    """Schedule workflows for the worker."""

    async with ExceptionGroupErrorLogging(logging.getLogger()):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                schedule_workflow(
                    client,
                    "sync-label-studio-laser-labels-workflow-schedule",
                    SyncLabelStudioLaserLabelsWorkflow,
                    timedelta(hours=1),
                )
            )
            tg.create_task(
                schedule_workflow(
                    client,
                    "sync-label-studio-headtail-labels-workflow-schedule",
                    SyncLabelStudioHeadTailLabelsWorkflow,
                    timedelta(hours=1),
                )
            )
            tg.create_task(
                schedule_workflow(
                    client,
                    "update-dashboard-config-workflow-schedule",
                    UpdateDashboardConfigWorkflow,
                    timedelta(hours=1),
                )
            )


async def main():
    """Main entry point for the worker."""

    configure_logging()
    log = logging.getLogger()

    tls_config = build_tls_config(settings.temporal)

    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}", tls=tls_config
    )

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE_NAME,
            workflows=[
                SyncLabelStudioLaserLabelsWorkflow,
                SyncLabelStudioHeadTailLabelsWorkflow,
                UpdateDashboardConfigWorkflow,
            ],
            activity_executor=executor,
            activities=[
                get_label_studio_projects_activity,
                get_laser_label_studio_project_ids_activity,
                get_headtail_label_studio_project_ids_activity,
                sync_laser_labels_for_label_studio_project_activity,
                sync_headtail_labels_for_label_studio_project_activity,
                sync_users_label_studio_activity,
                write_dashboard_config_activity,
            ],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

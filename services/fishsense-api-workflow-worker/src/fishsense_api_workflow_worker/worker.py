"""Worker for FishSense API Workflow"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Callable

from fishsense_shared import (
    ExceptionGroupErrorLogging,
    build_tls_config,
    ensure_schedule,
)
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleSpec,
    ScheduleState,
)
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.create_dive_slate_label_studio_project_activity import (  # pylint: disable=line-too-long
    create_dive_slate_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.create_headtail_label_studio_project_activity import (  # pylint: disable=line-too-long
    create_headtail_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.create_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
    create_laser_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.create_species_label_studio_project_activity import (  # pylint: disable=line-too-long
    create_species_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.get_active_label_studio_project_ids_activities import (  # pylint: disable=line-too-long
    get_active_dive_slate_label_studio_project_ids_activity,
    get_active_headtail_label_studio_project_ids_activity,
    get_active_laser_label_studio_project_ids_activity,
    get_active_species_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_dive_slate_label_studio_project_ids_activity import (  # pylint: disable=line-too-long
    get_dive_slate_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_headtail_label_studio_project_ids_activity import (  # pylint: disable=line-too-long
    get_headtail_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_label_studio_projects_activity import (
    get_label_studio_projects_activity,
)
from fishsense_api_workflow_worker.activities.get_laser_label_studio_project_ids_activity import (
    get_laser_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.populate_dive_slate_label_studio_project_activity import (  # pylint: disable=line-too-long
    populate_dive_slate_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.populate_headtail_label_studio_project_activity import (  # pylint: disable=line-too-long
    populate_headtail_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.populate_laser_label_studio_project_activity import (  # pylint: disable=line-too-long
    populate_laser_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.populate_species_label_studio_project_activity import (  # pylint: disable=line-too-long
    populate_species_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.sync_dive_slate_labels_for_label_studio_project_activity import (  # pylint: disable=line-too-long
    sync_dive_slate_labels_for_label_studio_project_activity,
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
from fishsense_api_workflow_worker.config import configure_logging, settings
from fishsense_api_workflow_worker.workflows.create_dive_slate_label_studio_project_workflow import (  # pylint: disable=line-too-long
    CreateDiveSlateLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.create_headtail_label_studio_project_workflow import (  # pylint: disable=line-too-long
    CreateHeadTailLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.create_laser_label_studio_project_workflow import (  # pylint: disable=line-too-long
    CreateLaserLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.create_species_label_studio_project_workflow import (  # pylint: disable=line-too-long
    CreateSpeciesLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_dive_slate_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateDiveSlateLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_headtail_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateHeadTailLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_laser_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateLaserLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.populate_species_label_studio_project_workflow import (  # pylint: disable=line-too-long
    PopulateSpeciesLabelStudioProjectWorkflow,
)
from fishsense_api_workflow_worker.workflows.sync_label_studio_dive_slate_labels_workflow import (
    SyncLabelStudioDiveSlateLabelsWorkflow,
)
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


async def schedule_workflow(
    client: Client, schedule_id: str, workflow_cls: Callable, interval: timedelta
):
    """Schedule a workflow to run periodically.

    Idempotent — uses the shared `ensure_schedule` helper, which treats
    `ScheduleAlreadyRunningError` as success and refuses to update an
    existing schedule in-place.
    """
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow_cls.run,
            args=(),
            id=f"{workflow_cls.__name__}-workflow",
            task_queue=TASK_QUEUE_NAME,
            # Sized to cover the worst-case sync run: 4 per-project sync
            # activities in parallel, each capped at 2h schedule_to_close
            # for first-run-on-backlog projects (see sync_label_studio_*
            # workflows). 3h leaves margin for the users + project-id
            # activities ahead of the per-project fan-out.
            run_timeout=timedelta(hours=3),
        ),
        spec=ScheduleSpec(intervals=[ScheduleIntervalSpec(every=interval)]),
        state=ScheduleState(),
    )
    await ensure_schedule(client, schedule_id=schedule_id, schedule=schedule)


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
                    "sync-label-studio-dive-slate-labels-workflow-schedule",
                    SyncLabelStudioDiveSlateLabelsWorkflow,
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
                SyncLabelStudioDiveSlateLabelsWorkflow,
                UpdateDashboardConfigWorkflow,
                CreateLaserLabelStudioProjectWorkflow,
                CreateSpeciesLabelStudioProjectWorkflow,
                CreateHeadTailLabelStudioProjectWorkflow,
                CreateDiveSlateLabelStudioProjectWorkflow,
                PopulateLaserLabelStudioProjectWorkflow,
                PopulateSpeciesLabelStudioProjectWorkflow,
                PopulateHeadTailLabelStudioProjectWorkflow,
                PopulateDiveSlateLabelStudioProjectWorkflow,
            ],
            activity_executor=executor,
            activities=[
                get_label_studio_projects_activity,
                get_laser_label_studio_project_ids_activity,
                get_headtail_label_studio_project_ids_activity,
                get_dive_slate_label_studio_project_ids_activity,
                sync_laser_labels_for_label_studio_project_activity,
                sync_headtail_labels_for_label_studio_project_activity,
                sync_dive_slate_labels_for_label_studio_project_activity,
                sync_users_label_studio_activity,
                write_dashboard_config_activity,
                create_laser_label_studio_project_activity,
                create_species_label_studio_project_activity,
                create_headtail_label_studio_project_activity,
                create_dive_slate_label_studio_project_activity,
                get_active_laser_label_studio_project_ids_activity,
                get_active_species_label_studio_project_ids_activity,
                get_active_headtail_label_studio_project_ids_activity,
                get_active_dive_slate_label_studio_project_ids_activity,
                populate_laser_label_studio_project_activity,
                populate_species_label_studio_project_activity,
                populate_headtail_label_studio_project_activity,
                populate_dive_slate_label_studio_project_activity,
            ],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

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

from fishsense_data_processing_workflow_worker.activities.cluster_dive_frames import (
    cluster_dive_frames,
)
from fishsense_data_processing_workflow_worker.activities.measure_fish_activity import (
    measure_fish_activity,
)
from fishsense_data_processing_workflow_worker.activities.perform_laser_calibration_activity import (  # noqa: E501  pylint: disable=line-too-long
    perform_laser_calibration_activity,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_dive_image import (
    preprocess_dive_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_headtail_image import (
    preprocess_headtail_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_laser_image import (
    preprocess_laser_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_slate_image import (
    preprocess_slate_image,
)
from fishsense_data_processing_workflow_worker.activities.resolve_laser_preprocess_inputs_activity import (  # noqa: E501  pylint: disable=line-too-long
    resolve_laser_preprocess_inputs_activity,
)
from fishsense_data_processing_workflow_worker.activities.select_next_high_priority_dive_for_laser_preprocessing_activity import (  # noqa: E501  pylint: disable=line-too-long
    select_next_high_priority_dive_for_laser_preprocessing_activity,
)
from fishsense_data_processing_workflow_worker.config import configure_logging, settings
from fishsense_data_processing_workflow_worker.workflows.dive_frame_clustering_workflow import (
    DiveFrameClusteringWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.measure_fish_workflow import (
    MeasureFishWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.perform_laser_calibration_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PerformLaserCalibrationWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_dive_images_workflow import (
    PreprocessDiveImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_headtail_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessHeadtailImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (
    PreprocessLaserImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_slate_images_workflow import (
    PreprocessSlateImagesWorkflow,
)

TASK_QUEUE_NAME = "fishsense_data_processing_queue"


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
            # One stage-0.1 invocation drains a single dive: an N-image
            # dive at ~2s/image rectify is well under an hour. 1h leaves
            # room for slow file-exchange roundtrips on a backed-up host.
            run_timeout=timedelta(hours=1),
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
                    "preprocess-laser-images-workflow-schedule",
                    PreprocessLaserImagesWorkflow,
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
                DiveFrameClusteringWorkflow,
                MeasureFishWorkflow,
                PerformLaserCalibrationWorkflow,
                PreprocessDiveImagesWorkflow,
                PreprocessHeadtailImagesWorkflow,
                PreprocessLaserImagesWorkflow,
                PreprocessSlateImagesWorkflow,
            ],
            activity_executor=executor,
            activities=[
                cluster_dive_frames,
                measure_fish_activity,
                perform_laser_calibration_activity,
                preprocess_dive_image,
                preprocess_headtail_image,
                preprocess_laser_image,
                preprocess_slate_image,
                resolve_laser_preprocess_inputs_activity,
                select_next_high_priority_dive_for_laser_preprocessing_activity,
            ],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

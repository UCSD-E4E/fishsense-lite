"""Data-worker Temporal worker construction and run loop.

Single ``build_worker`` construction point (shared with tests) plus the
``main`` scale-to-zero-friendly run loop. Caps activity concurrency to keep
the CPU-heavy per-image rectify/decode work under the pod's memory limit.
"""

import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from fishsense_shared import build_tls_config, temporal_namespace
from temporalio.client import Client
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
from fishsense_data_processing_workflow_worker.activities.preprocess_species_image import (
    preprocess_species_image,
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
from fishsense_data_processing_workflow_worker.activities.validate_laser_labels_for_dive_activity import (  # noqa: E501  pylint: disable=line-too-long
    validate_laser_labels_for_dive_activity,
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
from fishsense_data_processing_workflow_worker.workflows.preprocess_species_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessSpeciesImagesWorkflow,
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
from fishsense_data_processing_workflow_worker.workflows.validate_laser_labels_for_dive_workflow import (  # noqa: E501  pylint: disable=line-too-long
    ValidateLaserLabelsForDiveWorkflow,
)

TASK_QUEUE_NAME = "fishsense_data_processing_queue"

# How long in-flight activities get to finish when the worker is asked to
# stop. On NRP the api-worker scales this deployment to zero when idle, so
# a scale-down delivers SIGTERM mid-activity; without a graceful window the
# rectify/measure work in progress is cancelled immediately and re-queued
# (idempotent, so safe — this just avoids throwing away a nearly-done image).
# The k8s Deployment's terminationGracePeriodSeconds must be >= this.
GRACEFUL_SHUTDOWN_TIMEOUT = timedelta(seconds=30)

# Cap on activities executed at once. The per-image activities are async and
# offload full-res rawpy decode + opencv rectify via `asyncio.to_thread`, each
# peaking at ~1-2 GB. The Temporal SDK default (100) let a burst run ~8-9
# decodes concurrently, which blew the pod's 12 Gi limit → OOMKilled →
# CrashLoopBackOff (the whole worker died on startup within seconds, so nothing
# drained). Cap it low so peak memory stays bounded; tune via
# `general.max_concurrent_activities`.
DEFAULT_MAX_CONCURRENT_ACTIVITIES = 4


async def schedule_workflows(_: Client):
    """Schedule workflows for the worker.

    The data-worker owns no recurring schedules — those live on the
    always-up api-worker — so scaling this service to zero has no effect
    on schedule registration. Kept as a hook for symmetry with the other
    workers.
    """


def build_worker(
    client: Client,
    activity_executor: ThreadPoolExecutor,
    max_concurrent_activities: int = DEFAULT_MAX_CONCURRENT_ACTIVITIES,
) -> Worker:
    """Construct the data-worker Temporal worker.

    Single construction point so the worker config (workflows, activities,
    graceful-shutdown window, activity concurrency cap) is exercised by tests
    without standing up the full ``main`` loop.
    """
    return Worker(
        client,
        task_queue=TASK_QUEUE_NAME,
        max_concurrent_activities=max_concurrent_activities,
        workflows=[
            DiveFrameClusteringWorkflow,
            MeasureFishWorkflow,
            PerformLaserCalibrationWorkflow,
            PreprocessSpeciesImagesWorkflow,
            PreprocessHeadtailImagesWorkflow,
            PreprocessLaserImagesWorkflow,
            PreprocessSlateImagesWorkflow,
            ValidateLaserLabelsForDiveWorkflow,
        ],
        activity_executor=activity_executor,
        activities=[
            cluster_dive_frames,
            measure_fish_activity,
            perform_laser_calibration_activity,
            preprocess_species_image,
            preprocess_headtail_image,
            preprocess_laser_image,
            preprocess_slate_image,
            validate_laser_labels_for_dive_activity,
        ],
        graceful_shutdown_timeout=GRACEFUL_SHUTDOWN_TIMEOUT,
    )


async def main():
    """Main entry point for the worker."""

    configure_logging()
    log = logging.getLogger()

    tls_config = build_tls_config(settings.temporal)

    log.info(
        "connecting to Temporal host=%s:%d tls=%s",
        settings.temporal.host,
        settings.temporal.port,
        bool(tls_config),
    )
    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}",
        tls=tls_config,
        namespace=temporal_namespace(settings.temporal),
    )

    interrupt_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, interrupt_event.set)

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        async with build_worker(
            client,
            executor,
            settings.general.get(
                "max_concurrent_activities", DEFAULT_MAX_CONCURRENT_ACTIVITIES
            ),
        ):
            log.info("Worker started, scheduling workflows...")
            await schedule_workflows(client)
            await interrupt_event.wait()
            log.info(
                "shutdown signal received; draining "
                "(graceful_shutdown_timeout=%s)",
                GRACEFUL_SHUTDOWN_TIMEOUT,
            )


def run():
    """Run the worker."""
    asyncio.run(main())

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fishsense_shared import build_tls_config
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.cluster_dive_frames import (
    cluster_dive_frames,
)
from fishsense_data_processing_workflow_worker.config import configure_logging, settings
from fishsense_data_processing_workflow_worker.workflows.dive_frame_clustering_workflow import (
    DiveFrameClusteringWorkflow,
)

TASK_QUEUE_NAME = "fishsense_data_processing_queue"


async def schedule_workflows(_: Client):
    """Schedule workflows for the worker."""


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
            workflows=[DiveFrameClusteringWorkflow],
            activity_executor=executor,
            activities=[cluster_dive_frames],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

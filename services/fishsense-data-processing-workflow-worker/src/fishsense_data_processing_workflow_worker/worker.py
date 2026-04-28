import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fishsense_shared import build_tls_config
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.cluster_dive_frames import (
    cluster_dive_frames,
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
from fishsense_data_processing_workflow_worker.config import configure_logging, settings
from fishsense_data_processing_workflow_worker.workflows.dive_frame_clustering_workflow import (
    DiveFrameClusteringWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_dive_images_workflow import (
    PreprocessDiveImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_headtail_images_workflow import (
    PreprocessHeadtailImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (
    PreprocessLaserImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_slate_images_workflow import (
    PreprocessSlateImagesWorkflow,
)

TASK_QUEUE_NAME = "fishsense_data_processing_queue"


def collect_registrations(*, new_preprocess_workflows_enabled: bool):
    """Return `(workflows, activities)` for the worker, gated on the
    `feature_flags.new_preprocess_workflows` flag.

    When the flag is off, only the legacy `DiveFrameClusteringWorkflow`
    is registered. Workflow types that aren't registered with this
    worker will be created server-side by `start_workflow` calls but
    will never make progress, effectively blocking them.
    """
    workflows = [DiveFrameClusteringWorkflow]
    activities = [cluster_dive_frames]
    if new_preprocess_workflows_enabled:
        workflows += [
            PreprocessDiveImagesWorkflow,
            PreprocessHeadtailImagesWorkflow,
            PreprocessLaserImagesWorkflow,
            PreprocessSlateImagesWorkflow,
        ]
        activities += [
            preprocess_dive_image,
            preprocess_headtail_image,
            preprocess_laser_image,
            preprocess_slate_image,
        ]
    return workflows, activities


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

    new_preprocess_enabled = bool(settings.feature_flags.new_preprocess_workflows)
    if not new_preprocess_enabled:
        log.warning(
            "feature_flags.new_preprocess_workflows is OFF — stages 0.1, 2, "
            "5.1, and 9 will NOT be registered. Set "
            "E4EFS_FEATURE_FLAGS__NEW_PREPROCESS_WORKFLOWS=true to enable."
        )
    workflows, activities = collect_registrations(
        new_preprocess_workflows_enabled=new_preprocess_enabled,
    )

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE_NAME,
            workflows=workflows,
            activity_executor=executor,
            activities=activities,
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

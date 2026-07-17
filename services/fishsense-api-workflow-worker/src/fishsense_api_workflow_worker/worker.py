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
    temporal_namespace,
)
from temporalio.client import (
    Client,
    Schedule,
    ScheduleActionStartWorkflow,
    ScheduleIntervalSpec,
    ScheduleOverlapPolicy,
    SchedulePolicy,
    ScheduleSpec,
    ScheduleState,
)
from temporalio.worker import Worker

from fishsense_api_workflow_worker.activities.cleanup_raw_bytes_for_dive_activity import (  # pylint: disable=line-too-long
    cleanup_raw_bytes_for_dive_activity,
)
from fishsense_api_workflow_worker.activities.ensure_data_worker_running_activity import (  # pylint: disable=line-too-long
    ensure_data_worker_running_activity,
)
from fishsense_api_workflow_worker.activities.scale_down_data_worker_if_idle_activity import (  # pylint: disable=line-too-long
    scale_down_data_worker_if_idle_activity,
)
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
from fishsense_api_workflow_worker.activities.get_dive_slate_label_studio_project_ids_activity import (  # pylint: disable=line-too-long
    get_dive_slate_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_dives_with_complete_laser_labeling_activity import (  # pylint: disable=line-too-long
    get_dives_with_complete_laser_labeling_activity,
)
from fishsense_api_workflow_worker.activities.get_headtail_label_studio_project_ids_activity import (  # pylint: disable=line-too-long
    get_headtail_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_laser_label_studio_project_ids_activity import (
    get_laser_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.get_species_label_studio_project_ids_activity import (  # pylint: disable=line-too-long
    get_species_label_studio_project_ids_activity,
)
from fishsense_api_workflow_worker.activities.persist_dive_frame_clusters_activity import (  # pylint: disable=line-too-long
    persist_dive_frame_clusters_activity,
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
from fishsense_api_workflow_worker.activities.resolve_dive_frame_clustering_inputs_activity import (  # pylint: disable=line-too-long
    resolve_dive_frame_clustering_inputs_activity,
)
from fishsense_api_workflow_worker.activities.resolve_species_preprocess_inputs_activity import (  # pylint: disable=line-too-long
    resolve_species_preprocess_inputs_activity,
)
from fishsense_api_workflow_worker.activities.resolve_headtail_preprocess_inputs_activity import (  # pylint: disable=line-too-long
    resolve_headtail_preprocess_inputs_activity,
)
from fishsense_api_workflow_worker.activities.resolve_laser_preprocess_inputs_activity import (  # pylint: disable=line-too-long
    resolve_laser_preprocess_inputs_activity,
)
from fishsense_api_workflow_worker.activities.resolve_slate_preprocess_inputs_activity import (  # pylint: disable=line-too-long
    resolve_slate_preprocess_inputs_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_clustering_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_clustering_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_species_preprocessing_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_species_preprocessing_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_headtail_preprocessing_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_headtail_preprocessing_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_laser_calibration_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_laser_calibration_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_laser_preprocessing_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_laser_preprocessing_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_measure_fish_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_measure_fish_activity,
)
from fishsense_api_workflow_worker.activities.select_next_high_priority_dive_for_slate_preprocessing_activity import (  # pylint: disable=line-too-long
    select_next_high_priority_dive_for_slate_preprocessing_activity,
)
from fishsense_api_workflow_worker.activities.stage_raw_bytes_for_dive_activity import (  # pylint: disable=line-too-long
    stage_raw_bytes_for_dive_activity,
)
from fishsense_api_workflow_worker.activities.stage_slate_pdf_activity import (
    stage_slate_pdf_activity,
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
from fishsense_api_workflow_worker.activities.sync_species_labels_for_label_studio_project_activity import (  # pylint: disable=line-too-long
    sync_species_labels_for_label_studio_project_activity,
)
from fishsense_api_workflow_worker.activities.sync_users_label_studio_activity import (
    sync_users_label_studio_activity,
)
from fishsense_api_workflow_worker.activities.update_dive_image_groups_activity import (
    update_dive_image_groups_activity,
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
from fishsense_api_workflow_worker.workflows.cluster_dive_frames_parent_workflow import (  # pylint: disable=line-too-long
    ClusterDiveFramesParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.measure_fish_parent_workflow import (  # pylint: disable=line-too-long
    MeasureFishParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.perform_laser_calibration_parent_workflow import (  # pylint: disable=line-too-long
    PerformLaserCalibrationParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.preprocess_species_images_parent_workflow import (  # pylint: disable=line-too-long
    PreprocessSpeciesImagesParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.preprocess_headtail_images_parent_workflow import (  # pylint: disable=line-too-long
    PreprocessHeadtailImagesParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.preprocess_laser_images_parent_workflow import (  # pylint: disable=line-too-long
    PreprocessLaserImagesParentWorkflow,
)
from fishsense_api_workflow_worker.workflows.preprocess_slate_images_parent_workflow import (  # pylint: disable=line-too-long
    PreprocessSlateImagesParentWorkflow,
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
from fishsense_api_workflow_worker.workflows.sync_label_studio_species_labels_workflow import (  # pylint: disable=line-too-long
    SyncLabelStudioSpeciesLabelsWorkflow,
)
from fishsense_api_workflow_worker.workflows.update_dive_image_groups_workflow import (
    UpdateDiveImageGroupsWorkflow,
)
from fishsense_api_workflow_worker.workflows.scale_down_idle_data_worker_workflow import (  # pylint: disable=line-too-long
    ScaleDownIdleDataWorkerWorkflow,
)

TASK_QUEUE_NAME = "fishsense_api_queue"


async def schedule_workflow(
    client: Client,
    schedule_id: str,
    workflow_cls: Callable,
    interval: timedelta,
    *,
    offset: timedelta | None = None,
    run_timeout: timedelta = timedelta(hours=3),
    overlap: ScheduleOverlapPolicy = ScheduleOverlapPolicy.ALLOW_ALL,
):
    """Schedule a workflow to run periodically.

    Idempotent — uses the shared `ensure_schedule` helper, which treats
    `ScheduleAlreadyRunningError` as success and refuses to update an
    existing schedule in-place.

    Default `run_timeout` (3h) is sized for the worst-case label-studio
    sync run: 4 per-project sync activities in parallel, each capped
    at 2h schedule_to_close for first-run-on-backlog projects, plus
    margin for users + project-id activities. Workflows with shorter
    bounded work should override.

    `overlap=ALLOW_ALL` matches the long-standing behavior of the
    label-studio sync schedules. Workflows that read mutable shared
    state and need to be exclusive of themselves (e.g. selectors that
    pick the next dive in a queue) should pass `overlap=SKIP`.

    `offset` shifts the firing point within the interval — useful for
    spreading multiple hourly schedules across the hour so they don't
    all hit fishsense-api at the same moment.
    """
    interval_spec = (
        ScheduleIntervalSpec(every=interval, offset=offset)
        if offset is not None
        else ScheduleIntervalSpec(every=interval)
    )
    schedule = Schedule(
        action=ScheduleActionStartWorkflow(
            workflow_cls.run,
            args=(),
            id=f"{workflow_cls.__name__}-workflow",
            task_queue=TASK_QUEUE_NAME,
            run_timeout=run_timeout,
        ),
        spec=ScheduleSpec(intervals=[interval_spec]),
        policy=SchedulePolicy(overlap=overlap),
        state=ScheduleState(),
    )
    await ensure_schedule(client, schedule_id=schedule_id, schedule=schedule)


async def schedule_workflows(client: Client):
    """Schedule workflows for the worker."""

    log = logging.getLogger(__name__)
    log.info("registering Temporal schedules")
    async with ExceptionGroupErrorLogging(log):
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
                    "sync-label-studio-species-labels-workflow-schedule",
                    SyncLabelStudioSpeciesLabelsWorkflow,
                    timedelta(hours=1),
                )
            )
            # The four cross-worker preprocess parents run hourly with
            # SKIP-on-overlap (cluster safety: prevents two selectors
            # picking the same dive_id concurrently). Stagger by 15 min
            # so their selectors don't all hit `dives.get()` at the same
            # instant on the top of the hour. Stage-2 child can run
            # ~2h on a deep cluster; the others fit in 1h.
            #
            # Stage 1 (clustering) cascades from valid laser labels
            # (same trigger as 5.1) and lands PREDICTION clusters that
            # stage 2 then consumes. Slotted at +5 min so it has a
            # head start on the +15 stage-2 firing — clustering on a
            # ~hundred-image dive completes in seconds, so a single
            # +5/+15 schedule pair clears the laser→clustering→species
            # chain in one hour. SKIP overlap: same selector-race guard.
            tg.create_task(
                schedule_workflow(
                    client,
                    "cluster-dive-frames-workflow-schedule",
                    ClusterDiveFramesParentWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=5),
                    run_timeout=timedelta(minutes=30),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            tg.create_task(
                schedule_workflow(
                    client,
                    "preprocess-laser-images-workflow-schedule",
                    PreprocessLaserImagesParentWorkflow,
                    timedelta(hours=1),
                    run_timeout=timedelta(hours=1),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            tg.create_task(
                schedule_workflow(
                    client,
                    "preprocess-species-images-workflow-schedule",
                    PreprocessSpeciesImagesParentWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=15),
                    run_timeout=timedelta(hours=2),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            tg.create_task(
                schedule_workflow(
                    client,
                    "preprocess-headtail-images-workflow-schedule",
                    PreprocessHeadtailImagesParentWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=30),
                    run_timeout=timedelta(hours=1),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            tg.create_task(
                schedule_workflow(
                    client,
                    "preprocess-slate-images-workflow-schedule",
                    PreprocessSlateImagesParentWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=45),
                    run_timeout=timedelta(hours=1),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            # Stage 13 calibration parent: hourly, slotted at +50 min so
            # its selector doesn't collide with the four preprocess
            # parents at +0/+15/+30/+45. Calibration is pure math (no
            # NAS/exchange) so a 15-minute child execution_timeout is
            # plenty.
            tg.create_task(
                schedule_workflow(
                    client,
                    "perform-laser-calibration-workflow-schedule",
                    PerformLaserCalibrationParentWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=50),
                    run_timeout=timedelta(minutes=30),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            # Stage 14 measurement: hourly at +40 min, in the gap between
            # the headtail (+30) and slate (+45) parents. Scheduled as of
            # 2026-07-17 — it was operator-only while `post_measurement`
            # was a plain POST (a re-run duplicated measurements) and
            # while the cohort predicate could never go false (a schedule
            # would have re-measured the same dives every hour forever).
            # Both are fixed, and the cohort now drains to empty.
            #
            # Not slotted after calibration (+50) despite depending on it:
            # that would leave <5 min before the +55 scale-down sweeper.
            # A dive calibrated at :50 is instead measured at :40 the
            # following hour, which is irrelevant at this pipeline's
            # cadence — calibration is one-shot per dive and dives sit for
            # days. Each run drains exactly one dive.
            tg.create_task(
                schedule_workflow(
                    client,
                    "measure-fish-workflow-schedule",
                    MeasureFishParentWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=40),
                    # Child `execution_timeout` is 1h; add margin for the
                    # selector + data-worker scale-up activities.
                    run_timeout=timedelta(hours=1, minutes=30),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
            # Scale-to-zero sweeper for the NRP data-worker: hourly at
            # +55 min, after the last preprocess/calibration parent
            # firing, so it never races a parent that's still scaling
            # the data-worker *up*. SKIP overlap keeps a slow run from
            # stacking; a no-op when k8s scaling isn't configured.
            tg.create_task(
                schedule_workflow(
                    client,
                    "scale-down-idle-data-worker-workflow-schedule",
                    ScaleDownIdleDataWorkerWorkflow,
                    timedelta(hours=1),
                    offset=timedelta(minutes=55),
                    run_timeout=timedelta(minutes=10),
                    overlap=ScheduleOverlapPolicy.SKIP,
                )
            )
    log.info("Temporal schedules registered")


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

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE_NAME,
            workflows=[
                SyncLabelStudioLaserLabelsWorkflow,
                SyncLabelStudioHeadTailLabelsWorkflow,
                SyncLabelStudioDiveSlateLabelsWorkflow,
                SyncLabelStudioSpeciesLabelsWorkflow,
                CreateLaserLabelStudioProjectWorkflow,
                CreateSpeciesLabelStudioProjectWorkflow,
                CreateHeadTailLabelStudioProjectWorkflow,
                CreateDiveSlateLabelStudioProjectWorkflow,
                PopulateLaserLabelStudioProjectWorkflow,
                PopulateSpeciesLabelStudioProjectWorkflow,
                PopulateHeadTailLabelStudioProjectWorkflow,
                PopulateDiveSlateLabelStudioProjectWorkflow,
                UpdateDiveImageGroupsWorkflow,
                ClusterDiveFramesParentWorkflow,
                PreprocessLaserImagesParentWorkflow,
                PreprocessSpeciesImagesParentWorkflow,
                PreprocessHeadtailImagesParentWorkflow,
                PreprocessSlateImagesParentWorkflow,
                PerformLaserCalibrationParentWorkflow,
                MeasureFishParentWorkflow,
                ScaleDownIdleDataWorkerWorkflow,
            ],
            activity_executor=executor,
            activities=[
                get_laser_label_studio_project_ids_activity,
                get_headtail_label_studio_project_ids_activity,
                get_dive_slate_label_studio_project_ids_activity,
                get_dives_with_complete_laser_labeling_activity,
                get_species_label_studio_project_ids_activity,
                sync_laser_labels_for_label_studio_project_activity,
                sync_headtail_labels_for_label_studio_project_activity,
                sync_dive_slate_labels_for_label_studio_project_activity,
                sync_species_labels_for_label_studio_project_activity,
                sync_users_label_studio_activity,
                create_laser_label_studio_project_activity,
                create_species_label_studio_project_activity,
                create_headtail_label_studio_project_activity,
                create_dive_slate_label_studio_project_activity,
                populate_laser_label_studio_project_activity,
                populate_species_label_studio_project_activity,
                populate_headtail_label_studio_project_activity,
                populate_dive_slate_label_studio_project_activity,
                update_dive_image_groups_activity,
                resolve_dive_frame_clustering_inputs_activity,
                resolve_laser_preprocess_inputs_activity,
                resolve_species_preprocess_inputs_activity,
                resolve_headtail_preprocess_inputs_activity,
                resolve_slate_preprocess_inputs_activity,
                persist_dive_frame_clusters_activity,
                select_next_high_priority_dive_for_clustering_activity,
                select_next_high_priority_dive_for_laser_preprocessing_activity,
                select_next_high_priority_dive_for_species_preprocessing_activity,
                select_next_high_priority_dive_for_headtail_preprocessing_activity,
                select_next_high_priority_dive_for_slate_preprocessing_activity,
                select_next_high_priority_dive_for_laser_calibration_activity,
                select_next_high_priority_dive_for_measure_fish_activity,
                stage_raw_bytes_for_dive_activity,
                stage_slate_pdf_activity,
                cleanup_raw_bytes_for_dive_activity,
                ensure_data_worker_running_activity,
                scale_down_data_worker_if_idle_activity,
            ],
        )

        worker_task = worker.run()
        log.info("Worker started, scheduling workflows...")

        await schedule_workflows(client)
        await worker_task


def run():
    """Run the worker."""
    asyncio.run(main())

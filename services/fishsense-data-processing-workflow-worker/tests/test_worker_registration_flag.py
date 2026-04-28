"""Feature-flag gate on the new preprocess workflow registrations.

The four notebook ports (stages 0.1, 2, 5.1, 9) ship behind a single
config flag so they can be registered with the production worker only
once their api-worker drivers exist and the math has been re-verified
on real frames. Default is OFF so a deploy of this service binary
doesn't accidentally start servicing those task types.
"""

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
from fishsense_data_processing_workflow_worker.worker import collect_registrations
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


_NEW_WORKFLOWS = {
    PreprocessDiveImagesWorkflow,
    PreprocessHeadtailImagesWorkflow,
    PreprocessLaserImagesWorkflow,
    PreprocessSlateImagesWorkflow,
}
_NEW_ACTIVITIES = {
    preprocess_dive_image,
    preprocess_headtail_image,
    preprocess_laser_image,
    preprocess_slate_image,
}


def test_disabled_registers_only_legacy_clustering():
    workflows, activities = collect_registrations(
        new_preprocess_workflows_enabled=False
    )
    assert workflows == [DiveFrameClusteringWorkflow]
    assert activities == [cluster_dive_frames]


def test_disabled_excludes_every_new_workflow_and_activity():
    workflows, activities = collect_registrations(
        new_preprocess_workflows_enabled=False
    )
    assert _NEW_WORKFLOWS.isdisjoint(workflows)
    assert _NEW_ACTIVITIES.isdisjoint(activities)


def test_enabled_registers_legacy_plus_all_new():
    workflows, activities = collect_registrations(
        new_preprocess_workflows_enabled=True
    )
    assert DiveFrameClusteringWorkflow in workflows
    assert _NEW_WORKFLOWS.issubset(workflows)
    assert cluster_dive_frames in activities
    assert _NEW_ACTIVITIES.issubset(activities)


def test_no_duplicate_registrations_when_enabled():
    workflows, activities = collect_registrations(
        new_preprocess_workflows_enabled=True
    )
    assert len(workflows) == len(set(workflows))
    assert len(activities) == len(set(activities))

"""Activity to idempotently create the laser-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    create_or_get_label_studio_project,
)

LASER_PROJECT_TITLE = "FishSense — Laser Calibration Labeling (Stage 0.3)"

# Paste the labeling-config XML from your existing prod LS project
# (Project Settings -> Labeling Interface -> Code) here. Without it
# the activity raises rather than creating an unlabel-able project.
LASER_LABELING_CONFIG_XML = ""


@activity.defn
async def create_laser_label_studio_project_activity() -> int:
    """Create the laser-labeling LS project if it doesn't exist; return its ID.

    Idempotent — re-running returns the existing project's ID rather
    than creating a duplicate. Match is by title.
    """
    return await create_or_get_label_studio_project(
        project_title=LASER_PROJECT_TITLE,
        labeling_config_xml=LASER_LABELING_CONFIG_XML,
    )

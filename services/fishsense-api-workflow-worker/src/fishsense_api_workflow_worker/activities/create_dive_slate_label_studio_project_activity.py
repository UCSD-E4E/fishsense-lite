"""Activity to idempotently create the dive-slate-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    create_or_get_label_studio_project,
)

DIVE_SLATE_PROJECT_TITLE = "FishSense — Dive Slate Labeling (Stage 11)"

# Paste the labeling-config XML from your existing prod LS project
# (Project Settings -> Labeling Interface -> Code) here.
DIVE_SLATE_LABELING_CONFIG_XML = ""


@activity.defn
async def create_dive_slate_label_studio_project_activity() -> int:
    """Create the slate-labeling LS project if it doesn't exist; return its ID.

    Idempotent — re-running returns the existing project's ID rather
    than creating a duplicate. Match is by title.
    """
    return await create_or_get_label_studio_project(
        project_title=DIVE_SLATE_PROJECT_TITLE,
        labeling_config_xml=DIVE_SLATE_LABELING_CONFIG_XML,
    )

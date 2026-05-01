"""Activity to idempotently create the laser-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    create_or_get_label_studio_project,
)

LASER_PROJECT_TITLE = "FishSense — Laser Calibration Labeling (Stage 0.3)"

# Labeling-config XML from the prod laser project. The keypoint
# `from_name` is "laser" — must stay aligned with
# `LASER_LABEL_KEY_NAMES` in `sync_laser_labels_for_label_studio_project_activity.py`
# (currently `["kp-1", "laser"]`) so the sync side picks up
# annotations off the same control element this XML defines.
LASER_LABELING_CONFIG_XML = """\
<View>
  <KeyPointLabels name="laser" toName="img">
    <Label value="Red Laser" background="#FFDF20"/>
    <Label value="Green Laser" background="#A684FF"/>
  </KeyPointLabels>
  <Image name="img" value="$image" zoom="true" zoomControl="true"/>
</View>
"""


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

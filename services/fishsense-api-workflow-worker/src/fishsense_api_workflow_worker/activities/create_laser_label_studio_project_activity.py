"""Activity to idempotently create a per-dive laser-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_per_dive_title,
    create_or_get_label_studio_project,
)

LASER_PROJECT_TITLE_SUFFIX = "Laser Calibration Labeling"

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
async def create_laser_label_studio_project_activity(dive_id: int) -> int:
    """Create a per-dive laser-labeling LS project; return its ID.

    Title is `"{dive.name} - Laser Calibration Labeling"`. Idempotent
    — re-running for the same dive returns the existing project's ID
    rather than creating a duplicate. Match is by title.
    """
    title = await build_per_dive_title(dive_id, LASER_PROJECT_TITLE_SUFFIX)
    return await create_or_get_label_studio_project(
        project_title=title,
        labeling_config_xml=LASER_LABELING_CONFIG_XML,
    )

"""Activity to idempotently create the headtail-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    create_or_get_label_studio_project,
)

HEADTAIL_PROJECT_TITLE = "FishSense — HeadTail Labeling (Stage 5.3)"

# Labeling-config XML from the prod headtail project. The keypoint
# `from_name` is "kp-1" — must match the literal in
# `sync_headtail_labels_for_label_studio_project_activity.py` which
# filters annotations on `r["from_name"] == "kp-1"`.
HEADTAIL_LABELING_CONFIG_XML = """\
<View>
  <KeyPointLabels name="kp-1" toName="image">
    <Label value="Snout" background="#FFA39E"/>
    <Label value="Fork" background="#26a269"/>
  </KeyPointLabels>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
</View>
"""


@activity.defn
async def create_headtail_label_studio_project_activity() -> int:
    """Create the headtail-labeling LS project if it doesn't exist; return its ID.

    Idempotent — re-running returns the existing project's ID rather
    than creating a duplicate. Match is by title.
    """
    return await create_or_get_label_studio_project(
        project_title=HEADTAIL_PROJECT_TITLE,
        labeling_config_xml=HEADTAIL_LABELING_CONFIG_XML,
    )

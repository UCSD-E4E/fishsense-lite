"""Activity to idempotently create a per-dive headtail-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_per_dive_title,
    create_or_get_label_studio_project,
)

HEADTAIL_PROJECT_TITLE_SUFFIX = "HeadTail Labeling"

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
async def create_headtail_label_studio_project_activity(dive_id: int) -> int:
    """Create a per-dive headtail-labeling LS project; return its ID.

    Title is `"{dive.name} - HeadTail Labeling"`. Idempotent —
    re-running for the same dive returns the existing project's ID
    rather than creating a duplicate. Match is by title.
    """
    activity.logger.info(
        "create headtail LS project dive_id=%d", dive_id
    )
    title = await build_per_dive_title(dive_id, HEADTAIL_PROJECT_TITLE_SUFFIX)
    project_id = await create_or_get_label_studio_project(
        project_title=title,
        labeling_config_xml=HEADTAIL_LABELING_CONFIG_XML,
    )
    activity.logger.info(
        "create headtail LS project dive_id=%d project_id=%d title=%r",
        dive_id,
        project_id,
        title,
    )
    return project_id

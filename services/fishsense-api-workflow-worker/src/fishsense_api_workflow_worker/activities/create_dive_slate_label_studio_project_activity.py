"""Activity to idempotently create the dive-slate-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    create_or_get_label_studio_project,
)

DIVE_SLATE_PROJECT_TITLE = "FishSense — Dive Slate Labeling (Stage 11)"

# Labeling-config XML from the prod dive-slate project. Control names
# map 1:1 to fields on `DiveSlateLabel`:
#   * `upside_down` (Choices)        -> `DiveSlateLabel.upside_down`
#   * `reference_points` (KeyPoints) -> `DiveSlateLabel.reference_points`
#   * `slate` (RectangleLabels)      -> `DiveSlateLabel.slate_rectangle`
#   * `skipped_points` (TextArea)    -> `DiveSlateLabel.skipped_points`
# The stage 12 sync activity (sync_slate_label, not yet ported) will
# need to read annotations off these `from_name`s.
DIVE_SLATE_LABELING_CONFIG_XML = """\
<View>

  <Choices name="upside_down" toName="image">
    <Choice value="Slate upside down" />
  </Choices>

  <Image name="image" value="$image" zoom="true"/>

  <KeyPointLabels name="reference_points" toName="image">
    <Label value="Reference Point" background="red"/>
  </KeyPointLabels>

  <RectangleLabels name="slate" toName="image">
    <Label value="Slate" background="green" />
  </RectangleLabels>

  <Header value="Skipped points.  Use a comma separated list." />
  <TextArea name="skipped_points" toName="image"/>

</View>
"""


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

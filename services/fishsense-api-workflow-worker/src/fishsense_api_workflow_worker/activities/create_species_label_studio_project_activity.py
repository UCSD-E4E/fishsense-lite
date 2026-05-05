"""Activity to idempotently create a per-dive species-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_per_dive_title,
    create_or_get_label_studio_project,
)

SPECIES_PROJECT_TITLE_SUFFIX = "Species Labeling"

# Labeling-config XML from the prod species project. Several control
# names are load-bearing for downstream code:
#   * `name="laser"` — same name used by the laser project, so the
#     sync activity's `LASER_LABEL_KEY_NAMES = ["kp-1", "laser"]`
#     picks up laser keypoints labeled here too.
#   * `name="species"` taxonomy — its top-level `Slate` branch with
#     the `Laser on slate` leaf is what stage 11 keys on
#     (`SLATE_CONTENT_MARKER = "Slate, Laser on slate"` in
#     `populate_dive_slate_label_studio_project_activity.py`).
#   * `name="exclude"` with `Top 3 photos of group` — what
#     `populate_headtail_label_studio_project_activity.py`
#     filters on via `species_label.top_three_photos_of_group`.
# Renaming any of these breaks the downstream populate / sync chain
# silently (the activity returns no rows rather than erroring).
SPECIES_LABELING_CONFIG_XML = """\
<View>
  <Choices name="grouping" toName="image">
    <Choice value="Part of previous group" />
    <Choice value="Not part of current group" />
  </Choices>

  <Choices name="exclude" toName="image">
    <Choice value="Top 3 photos of group" />
  </Choices>

  <Choices name="slate" toName="image">
    <Choice value="Slate upside down" />
  </Choices>

  <Image name="image" value="$image"/>

  <Header value="Please label the laser" />
  <KeyPointLabels name="laser" toName="image">
    <Label value="Red Laser" background="#FFA39E"/>
    <Label value="Green Laser" background="#26a269"/>
  </KeyPointLabels>

  <Header value="Please select the content of the image" />
  <Taxonomy name="species" toName="image" leafsOnly="true">
    <Choice value="None"/>
    <Choice value="Slate">
      <Choice value="Laser on slate"/>
      <Choice value="Laser not on slate"/>
    </Choice>
    <Choice value="Fish">
      <Choice value="Hogfish (Lachnolaimus maximus)"/>
      <Choice value="Black Grouper (Mycteroperca bonaci)"/>
      <Choice value="Goliath Grouper (Epinephelus itajara)"/>
      <Choice value="Nassau Grouper (Epinephelus striatus)"/>
      <Choice value="Red Grouper (Epinephelus morio)"/>
      <Choice value="Yellowtail Snapper (Ocyurus chrysurus)"/>
      <Choice value="Grey Snapper (Lutjanus griseus)"/>
      <Choice value="Mutton Snapper (Lutjanus analis)"/>
      <Choice value="Blue Parrotfish (Scarus coeruleus)"/>
      <Choice value="Midnight Parrotfish (Scarus coelestinus)"/>
      <Choice value="Rainbow Parrotfish (Scarus guacamaia)"/>
      <Choice value="Stoplight Parrotfish (Sparisoma viride)"/>
      <Choice value="Yellowmouth Grouper (Mycteroperca interstitialis)"/>
      <Choice value="Unidentifiable (Cannot see)"/>
      <Choice value="Other (Identifiable but Nontarget)"/>
      <Choice value="Other (Fish Model)"/>
    </Choice>
  </Taxonomy>

  <Header value="Is the fish measurable?"/>
  <Taxonomy name="measurable" toName="image">
    <Choice value="yes, center of fish" />
    <Choice value="yes, not center of fish" />
    <Choice value="no" />
  </Taxonomy>

  <Header value="How angled is the fish?"/>
  <Taxonomy name="fishAngles" toName="image">
    <Choice value="x &lt; 5°" />
    <Choice value="5° &lt; x &lt; 10°" />
    <Choice value="10° &lt; x &lt; 15°" />
    <Choice value="x &gt; 15°" />
  </Taxonomy>

  <Header value="How curved is the fish?"/>
  <Taxonomy name="fishCurve" toName="image">
    <Choice value="No Curve" />
    <Choice value="Slight Curve" />
    <Choice value="Significant Curve" />
  </Taxonomy>
</View>
"""


@activity.defn
async def create_species_label_studio_project_activity(dive_id: int) -> int:
    """Create a per-dive species-labeling LS project; return its ID.

    Title is `"{dive.name} - Species Labeling"`. Idempotent —
    re-running for the same dive returns the existing project's ID
    rather than creating a duplicate. Match is by title.
    """
    title = await build_per_dive_title(dive_id, SPECIES_PROJECT_TITLE_SUFFIX)
    return await create_or_get_label_studio_project(
        project_title=title,
        labeling_config_xml=SPECIES_LABELING_CONFIG_XML,
    )

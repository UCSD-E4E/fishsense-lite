"""Activity to idempotently create a per-dive species-labeling LS project."""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    build_per_dive_title,
    create_or_get_label_studio_project,
)

SPECIES_PROJECT_TITLE_SUFFIX = "Species Labeling"

# Labeling-config XML supplied by the user 2026-05-05. Several control
# names are load-bearing for downstream code:
#   * `name="grouping"` choices — stage 6.1
#     (`update_dive_image_groups_activity`) walks PREDICTION clusters
#     using these labels to materialize LABEL_STUDIO clusters.
#   * `name="exclude"` with `Top 3 photos of group` — historically used
#     by stage 5.1 head/tail. The 2026-05-04 cascade flip moved
#     head/tail off this gate (it now cascades from valid lasers), so
#     this control is informational rather than load-bearing on the
#     species side, but kept in the XML to preserve labeler workflow.
#   * `name="species"` taxonomy — its top-level `Slate` branch with
#     the `Laser on slate` leaf is what stage 9 keys on
#     (`SLATE_CONTENT_MARKER = "Slate, Laser on slate"` in
#     `populate_dive_slate_label_studio_project_activity.py`).
# Renaming any of these breaks the downstream populate / sync chain
# silently (the activity returns no rows rather than erroring).
#
# Schema diffs vs the prior XML:
#   - `<KeyPointLabels name="laser">` removed: lasers are labeled in
#     their own dedicated project (stage 0.1).
#   - `<Choices name="slate">` ("Slate upside down") removed.
#   - Slate branch expanded with H-Slate, Tic-Tac-Toe 1..6, V-Slate 1..4.
#   - Fish Model branch added (George, Purple Angel, Purple Ant,
#     Yellow Ant, Yellow Anthias, Snook).
SPECIES_LABELING_CONFIG_XML = """\
<View>
  <Choices name="grouping" toName="image">
    <Choice value="Part of previous group" />
    <Choice value="Not part of current group" />
  </Choices>

  <Choices name="exclude" toName="image">
    <Choice value="Top 3 photos of group" />
  </Choices>

  <Image name="image" value="$image"/>

  <Header value="Please select the content of the image" />
  <Taxonomy name="species" toName="image" leafsOnly="true">
    <Choice value="None"/>
    <Choice value="Slate">
      <Choice value="Laser on slate"/>
      <Choice value="Laser not on slate"/>

      <Choice value="H-Slate"/>

      <Choice value="Tic-Tac-Toe 1"/>
      <Choice value="Tic-Tac-Toe 2"/>
      <Choice value="Tic-Tac-Toe 3"/>
      <Choice value="Tic-Tac-Toe 4"/>
      <Choice value="Tic-Tac-Toe 5"/>
      <Choice value="Tic-Tac-Toe 6"/>

      <Choice value="V-Slate 1"/>
      <Choice value="V-Slate 2"/>
      <Choice value="V-Slate 3"/>
      <Choice value="V-Slate 4"/>
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
    </Choice>
    <Choice value="Fish Model">
      <Choice value="George"/>
      <Choice value="Purple Angel"/>
	  <Choice value="Purple Ant"/>
      <Choice value="Yellow Ant"/>
	  <Choice value="Yellow Anthias"/>
      <Choice value="Snook"/>
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
    activity.logger.info(
        "create species LS project dive_id=%d", dive_id
    )
    title = await build_per_dive_title(dive_id, SPECIES_PROJECT_TITLE_SUFFIX)
    project_id = await create_or_get_label_studio_project(
        project_title=title,
        labeling_config_xml=SPECIES_LABELING_CONFIG_XML,
    )
    activity.logger.info(
        "create species LS project dive_id=%d project_id=%d title=%r",
        dive_id,
        project_id,
        title,
    )
    return project_id

"""Activity to idempotently create the checkerboard lattice-verification project.

**One project for every dive, which breaks this repo's per-dive convention on
purpose.** Every other stage gives each dive its own project so labelers can
track per-dive progress. This one is a study, not a labeling queue, and its
whole value depends on the labeler not knowing which calibration a frame came
from: the hypothesis under test predicts that faults concentrate in five
specific dives, so a per-dive project would tell the judge the answer before
they looked. One shuffled project keeps it blind, and the frames' dives are
recovered afterwards from the task's image checksum.

That also makes the title fixed rather than built by `build_per_dive_title`,
and the project correspondingly long-lived: re-running the study adds tasks to
the same project instead of standing up a new one, and
`import_tasks_and_record_labels` dedupes by URL so an unchanged frame is not
re-imported.
"""

from temporalio import activity

from fishsense_api_workflow_worker.activities.populate_utils import (
    create_or_get_label_studio_project,
)

CHECKERBOARD_LATTICE_PROJECT_TITLE = "Checkerboard Lattice Verification"

# The question is deliberately narrow, and phrased about *spacing* rather than
# about correctness in general. A labeler asked "is this detection good?" will
# report blur, glare and partial boards — all real, none of them the thing that
# scales a calibration. The fault this study exists to find is that consecutive
# marks sit two board squares apart instead of one.
#
# `lattice_verdict` is required; `lattice_fault` is not. A bare correct/wrong
# split would confirm that some fits are bad, which is already known from the
# baselines — it is the fault *kind* that distinguishes a coarse lattice from
# every other way a detection can be wrong, and so decides whether the leading
# explanation survives.
#
# The keypoints arrive as predictions (see
# `populate_checkerboard_lattice_label_studio_project_activity`) and are also
# burned into the JPEG with their connecting edges, because isolated dots make
# "every corner or every other one?" a counting exercise while drawn cells make
# it obvious. The KeyPointLabels control is here so the prediction has somewhere
# to land and can be toggled off against the underlying board.
CHECKERBOARD_LATTICE_LABELING_CONFIG_XML = """\
<View>

  <Header value="Does every marked point sit on a board corner, with none skipped?"/>

  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>

  <KeyPointLabels name="lattice" toName="image" opacity="0.9" strokewidth="3">
    <Label value="Detected corner" background="#FF3B30"/>
  </KeyPointLabels>

  <Header value="Verdict"/>
  <Choices name="lattice_verdict" toName="image" choice="single" required="true" showInLine="true">
    <Choice value="Correct"/>
    <Choice value="Incorrect"/>
    <Choice value="Cannot tell"/>
  </Choices>

  <Header value="If incorrect, what is wrong?"/>
  <Choices name="lattice_fault" toName="image" choice="single">
    <Choice value="Skips corners - marks are 2 or more squares apart"/>
    <Choice value="Denser than the squares"/>
    <Choice value="Marks are not on corners at all"/>
    <Choice value="Different board from the E4E 14x10"/>
  </Choices>

</View>
"""


@activity.defn
async def create_checkerboard_lattice_label_studio_project_activity() -> int:
    """Create the lattice-verification LS project; return its ID.

    Takes no `dive_id` — there is one project for the whole study. Idempotent:
    re-running returns the existing project's ID rather than creating a
    duplicate. Match is by title.
    """
    activity.logger.info("create checkerboard lattice-verification LS project")
    project_id = await create_or_get_label_studio_project(
        project_title=CHECKERBOARD_LATTICE_PROJECT_TITLE,
        labeling_config_xml=CHECKERBOARD_LATTICE_LABELING_CONFIG_XML,
    )
    activity.logger.info(
        "checkerboard lattice-verification LS project_id=%d title=%r",
        project_id,
        CHECKERBOARD_LATTICE_PROJECT_TITLE,
    )
    return project_id

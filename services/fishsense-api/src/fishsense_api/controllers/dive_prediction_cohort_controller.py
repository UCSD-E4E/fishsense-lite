# pylint: disable=C0121
#   `== True` / `!= None` are required here, not sloppy: SQLAlchemy overloads
#   the comparison operators to build SQL expressions, and `is True` / `is not
#   None` would evaluate to a Python bool and silently drop the predicate.
"""Cohort selectors for the model-assisted (prediction) stages.

Split out of `dive_cohort_controller` when that module passed 1000 lines — the
same threshold that caused it to be split out of `dive_controller`. These three
are a coherent group: each answers "which dive should a *detector* run on
next", and they share a shape the preprocess cohorts do not. An image needs a
prediction when it has neither been predicted nor labelled by a human, and each
also re-enters on a **version mismatch** rather than on absence, which is what
makes improving a detector a drainable cohort instead of a hand-run backfill.

**Route ordering.** This module is imported before `dive_controller`, matching
`dive_cohort_controller`. The usual justification is that `/dives/select-next/...`
must beat `/dives/{dive_id}` — but that was checked while splitting this out
rather than repeated: the catch-all compiles to
`^/api/v1/dives/(?P<dive_id>[^/]+)$`, anchored to one segment with no trailing
slash, so it cannot shadow a two-segment route ending in `/`. Reordering the
import does not break these three, and the guard passes either way.

The ordering is still kept, for two reasons worth stating rather than assuming:
it is the convention every `/dives/...` collection module follows, and a
selector added here later as a bare single segment would genuinely be at risk.
`test_dive_route_disambiguation.py` now covers all three of these routes — it
did not cover any prediction selector before this split.
"""

from typing import List

from fastapi import Depends
from fishsense_shared import (
    HEADTAIL_PREDICTOR_VERSION,
    LASER_PREDICTOR_VERSION,
    taxonomy,
)
from sqlalchemy import and_, or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.head_tail_prediction import HeadTailPrediction
from fishsense_api.models.image import Image
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.laser_prediction import LaserPrediction
from fishsense_api.models.priority import Priority
from fishsense_api.models.slate_prediction import SlatePrediction
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

# Stage-9 `species_label.content_of_image` marker. Imported from the shared
# taxonomy vocabulary rather than re-exported from `dive_cohort_controller`:
# CLAUDE.md is explicit that this marker is spelled once, in
# `fishsense_shared.taxonomy`, and never inline in a controller — it drifted
# across three copies before it lived there.
SLATE_CONTENT_MARKER = taxonomy.SLATE_CONTENT_MARKER


@app.get("/api/v1/dives/select-next/laser-prediction/")
async def select_next_for_laser_prediction(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Model-assisted laser labeling: HIGH-priority + at least one image
    with no `LaserPrediction` row and no *completed* `LaserLabel` row.

    An image needs a prediction only if it has neither been predicted nor
    *labeled by a human* yet — so a dive drops out of the cohort once every
    image is predicted, and images a human already labeled are never predicted
    over.

    Re-prediction is no longer "a manual affair": a dive is also selected while
    it is still being labeled and carries a prediction from an older
    `LASER_PREDICTOR_VERSION`.

    "Labeled" here means `completed IS TRUE`, NOT merely
    `project_id IS NOT NULL`: the laser populate step seeds placeholder
    rows (`completed=False`, x/y NULL) that *carry* a `project_id`, so a
    project-id check would exclude every populate-seeded-but-unlabeled
    image — starving the detector on exactly the dives it should assist
    (e.g. a dive populated before the detector shipped). Matches populate's
    own `completed`-based definition of "labeled" (`_select_unlabeled_images`).

    "Labeled" also requires `superseded IS FALSE`: a completed label that
    `ValidateLaserLabelsForDiveWorkflow`'s RANSAC pass dead-lettered is an
    *invalidated* label — the image has no live human label and should
    re-enter the cohort. Mirrors the superseded-filter every downstream read
    (`get_laser_labels`, the preprocess/predict resolvers) already applies.
    """
    has_image_needing_prediction = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(
            ~select(LaserPrediction.id)
            .where(LaserPrediction.image_id == Image.id)
            .exists()
        )
        .where(
            ~select(LaserLabel.id)
            .where(LaserLabel.image_id == Image.id)
            .where(LaserLabel.completed == True)
            .where(LaserLabel.superseded == False)
            .exists()
        )
        .exists()
    )
    # Second way in: a stale-version prediction on a dive still being labeled.
    # Why mismatch rather than absence, and why only actively-labeled dives:
    # `fishsense_shared.laser_predictor`. `IS DISTINCT FROM`, not `!=` — every
    # pre-versioning row is NULL, and `!=` would answer NULL and select nothing.
    version = LASER_PREDICTOR_VERSION
    dive_is_still_being_labeled = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(
            select(LaserLabel.id)
            .where(LaserLabel.image_id == Image.id)
            .where(LaserLabel.completed == False)
            .where(LaserLabel.superseded == False)
            .where(LaserLabel.label_studio_project_id != None)
            .exists()
        )
        .exists()
    )
    has_image_with_a_stale_prediction = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(
            select(LaserPrediction.id)
            .where(LaserPrediction.image_id == Image.id)
            # pylint: disable-next=no-member
            .where(LaserPrediction.predictor_version.is_distinct_from(version))
            .exists()
        )
        # Never re-predict over finished human work (the guard, not a fallout).
        .where(
            ~select(LaserLabel.id)
            .where(LaserLabel.image_id == Image.id)
            .where(LaserLabel.completed == True)
            .where(LaserLabel.superseded == False)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(
            or_(
                has_image_needing_prediction,
                and_(dive_is_still_being_labeled, has_image_with_a_stale_prediction),
            )
        )
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/headtail-prediction/")
async def select_next_for_headtail_prediction(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Model-assisted head/tail labeling: HIGH-priority + at least one
    canonical image with a *valid* `LaserLabel` that still needs predicting.

    The laser dot is not just a filter here, it is the crop centre — the
    predictor looks only at a window around it rather than searching the frame
    — so an image without a live dot has nothing for this stage to do.

    An image needs a prediction when it has no live human `HeadTailLabel` and
    either has no `HeadTailPrediction` at all, or has one that is *stale*.
    Stale has two forms, and both are mismatch rather than absence, which is
    what makes improving the stage or cleaning up lasers an ordinary drainable
    cohort instead of a hand-run backfill:

    * the row came from an older `HEADTAIL_PREDICTOR_VERSION`
      (`fishsense_shared.headtail_predictor` explains why the version is a
      hand-bumped literal and not a checkpoint hash); or
    * the row names a `laser_label_id` that has since been superseded — the
      dot that chose the fish was dead-lettered, so the mask may be of the
      wrong thing entirely. Same provenance idea as `LaserDepth`.

    `IS DISTINCT FROM`, not `!=`: pre-versioning rows are NULL and `!=` would
    answer NULL, selecting nothing.

    "Labelled" means `completed IS TRUE AND superseded IS FALSE`. A
    populate-seeded sentinel row is not a label, and a completed row a later
    pass dead-lettered leaves the image with no live human work, so it
    re-enters.

    Every subquery correlates explicitly. Auto-correlation only reaches the
    immediately enclosing SELECT, and an uncorrelated one compiles to a
    multi-row scalar subquery that Postgres rejects while SQLite silently
    answers with the first row — the shape that 500'd two selectors on every
    hourly poll on 2026-08-18.
    """
    version = HEADTAIL_PREDICTOR_VERSION

    has_live_laser = (
        select(LaserLabel.id)
        .where(LaserLabel.image_id == Image.id)
        .where(LaserLabel.completed == True)
        .where(LaserLabel.superseded == False)
        .where(LaserLabel.x != None)
        .where(LaserLabel.y != None)
        .correlate(Image)
        .exists()
    )
    has_live_headtail_label = (
        select(HeadTailLabel.id)
        .where(HeadTailLabel.image_id == Image.id)
        .where(HeadTailLabel.completed == True)
        .where(HeadTailLabel.superseded == False)
        .correlate(Image)
        .exists()
    )
    has_any_prediction = (
        select(HeadTailPrediction.id)
        .where(HeadTailPrediction.image_id == Image.id)
        .correlate(Image)
        .exists()
    )
    laser_behind_prediction_was_superseded = (
        select(LaserLabel.id)
        .where(LaserLabel.id == HeadTailPrediction.laser_label_id)
        .where(LaserLabel.superseded == True)
        .correlate(HeadTailPrediction)
        .exists()
    )
    has_stale_prediction = (
        select(HeadTailPrediction.id)
        .where(HeadTailPrediction.image_id == Image.id)
        .where(
            or_(
                # pylint: disable-next=no-member
                HeadTailPrediction.predictor_version.is_distinct_from(version),
                laser_behind_prediction_was_superseded,
            )
        )
        .correlate(Image)
        .exists()
    )
    has_image_needing_prediction = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(has_live_laser)
        .where(~has_live_headtail_label)
        .where(or_(~has_any_prediction, has_stale_prediction))
        .correlate(Dive)
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_image_needing_prediction)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/slate-prediction/")
async def select_next_for_slate_prediction(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Model-assisted slate labeling: HIGH-priority + `dive_slate_id` set + at
    least one slate frame with no `SlatePrediction` and no *completed*
    `DiveSlateLabel`.

    A slate frame is an image with a `SpeciesLabel.content_of_image =
    'Slate, Laser on slate'` (the same frames stage 9 preprocesses). Such a
    frame needs a prediction only if it has neither been predicted nor
    labeled by a human yet — so a dive drops out once every slate frame is
    predicted (one-shot per image), and human-labeled frames are never
    predicted over. "Labeled" = `completed IS TRUE AND superseded IS FALSE`,
    matching the laser-prediction cohort's rationale (populate seeds
    placeholder rows that carry a project_id, so a project-id check would
    starve the detector).
    """
    has_slate_frame_needing_prediction = (
        select(SpeciesLabel.id)
        .join(Image, Image.id == SpeciesLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(SpeciesLabel.content_of_image == SLATE_CONTENT_MARKER)
        .where(
            ~select(SlatePrediction.id)
            .where(SlatePrediction.image_id == Image.id)
            .exists()
        )
        .where(
            ~select(DiveSlateLabel.id)
            .where(DiveSlateLabel.image_id == Image.id)
            .where(DiveSlateLabel.completed == True)
            .where(DiveSlateLabel.superseded == False)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(Dive.dive_slate_id != None)
        .where(has_slate_frame_needing_prediction)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/needing-headtail-population/")
async def select_dives_needing_headtail_population(
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Dives that need model-assisted head/tail LS tasks (re)populated.

    Cohort: HIGH priority + at least one canonical image with a *valid* laser
    dot, a `HeadTailPrediction` (the detector has run for it), and no
    *completed* `HeadTailLabel`.

    The prediction gate is what makes this the decoupled populate cohort rather
    than the preprocess one: populate seeds sentinel `HeadTailLabel` rows, and
    the predict cohort excludes any image with a live label, so populating an
    unpredicted image would starve it of a prediction forever. Gating on
    "prediction exists" guarantees the detector ran first. An abstention still
    opens the gate — the image was visited, and withholding it would strand it
    with neither a prediction nor a human label.

    "No completed label" rather than "no row" keeps a dive in the cohort until
    its labelers finish, so the idempotent populate self-heals hourly. Returns
    every match; the scheduled parent fans out one child per dive.
    """
    has_populatable_image = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(
            select(LaserLabel.id)
            .where(LaserLabel.image_id == Image.id)
            .where(LaserLabel.completed == True)
            .where(LaserLabel.superseded == False)
            .where(LaserLabel.x != None)
            .where(LaserLabel.y != None)
            .correlate(Image)
            .exists()
        )
        .where(
            select(HeadTailPrediction.id)
            .where(HeadTailPrediction.image_id == Image.id)
            .correlate(Image)
            .exists()
        )
        .where(
            ~select(HeadTailLabel.id)
            .where(HeadTailLabel.image_id == Image.id)
            .where(HeadTailLabel.completed == True)
            .correlate(Image)
            .exists()
        )
        .correlate(Dive)
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_populatable_image)
        .order_by(Dive.id)
    )
    return list((await session.exec(query)).all())

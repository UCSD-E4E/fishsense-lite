# pylint: disable=C0121
#   `== True` / `!= None` are required here, not sloppy: SQLAlchemy overloads
#   the comparison operators to build SQL expressions, and `is True` / `is not
#   None` would evaluate to a Python bool and silently drop the predicate.
"""Cohort selectors — which dive each pipeline stage should work on next.

Split out of `dive_controller` because that module had grown past 1000 lines
and was carrying a `too-many-lines` disable. These endpoints are a coherent
group with a different job from dive CRUD: each answers "what is the next
HIGH-priority dive whose pipeline state matches stage N's cohort", and the
api-worker's hourly schedules poll them.

**Route ordering is load-bearing.** `/dives/select-next/...` must be registered
BEFORE `/dives/{dive_id}`, or FastAPI tries to coerce "select-next" into the
`{dive_id}: int` path param and 422s. Declaration order is registration order,
and across modules that means *import* order in `controllers/__init__.py`,
where this module deliberately precedes `dive_controller`.
`test_dive_route_disambiguation.py` is the guard.

Each cohort predicate here is mirrored by a flag in `dive_pipeline_status`
(`views.py`). The two are the same question asked in two languages and must
stay in step — a drift means the dashboard reports work the worker never does,
which is silent. `test_dive_pipeline_status_view.py` pins the agreement.
"""

import logging
from typing import List

from fastapi import Depends
from sqlalchemy import and_, func, or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_shared import taxonomy

from fishsense_api.database import get_async_session
from fishsense_api.models.data_source import DataSource
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_frame_cluster import (
    DiveFrameCluster,
    DiveFrameClusterImageMapping,
)
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.laser_depth import LaserDepth
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.laser_prediction import LaserPrediction
from fishsense_api.models.measurement import Measurement
from fishsense_api.models.priority import Priority
from fishsense_api.models.slate_prediction import SlatePrediction
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)


# Every cohort selector below correlates `Image` to `Dive` and then adds
# `Image.is_canonical == True`. That filter is not an optimisation -- it is
# what stops a duplicate dive wedging the pipeline.
#
# The same physical frames legitimately live under several dive rows (half of
# prod's image table is duplicate content; 207 of 479 dives are duplicates end
# to end), and `is_canonical` marks which copy is the real one. Without the
# filter, a duplicate dive promoted to HIGH satisfies "has an image with no
# label row" forever: populate declines to make LS tasks for frames that
# already exist elsewhere, so no label row appears, so the dive never drains --
# re-staging its raw `.ORF`s from the NAS every hour and blocking every
# higher-id dive behind it. That is the prod dive 60 failure.
#
# Duplicate dives are all LOW priority today, so this is currently latent. That
# is a property of the data, not of the code. `test_canonical_only_pipeline_work.py`
# pins both the behaviour and the "every selector has it" registry property.


def _valid_laser_conditions():
    """The repo-wide definition of a *valid* laser label.

    The labeler placed a point, the validator signed off, and
    `ValidateLaserLabelsForDiveWorkflow`'s RANSAC fit hasn't superseded
    it. Stages 1, 2, 5.1 and 14 all cascade from this gate, so it's
    spelled once. Splat into a select: `.where(*_valid_laser_conditions())`.

    Mirrors `views._VALID_LASER_SQL` — the view and these selectors are
    two representations of the same predicate and must stay in step.
    `test_dive_pipeline_status_view.py` pins that agreement.
    """
    return (
        LaserLabel.completed == True,
        LaserLabel.superseded == False,
        LaserLabel.x != None,
        LaserLabel.y != None,
    )


def _measurable_species_conditions():
    """A species row stage 14 can actually turn into a Measurement.

    Three measurable branches, matching `taxonomy.is_measurable`:

        "Fish, Hogfish (Lachnolaimus maximus)"  -> real fish, `Common
                                                   (Scientific)` leaf
        "Fish Model, Weasly Fish"               -> rigid model, name-keyed
        "Calibration Targets, Ruler"            -> the ruler, name-keyed

    Everything else — `"Slate, Laser on slate"`, other Calibration Targets —
    is not measurable. (An earlier version of this docstring listed the
    bottom two branches as *skipped*, six lines above the code matching them.
    It was written when only the first branch existed and never updated when
    models and the ruler were added; the ruler clause in particular looks
    like dead code if you believe the comment.)

    Without this condition the cohort and the activity disagree in a way that
    cannot resolve: the selector keeps offering an image the activity always
    skips, so no Measurement is written, `~is_measured` stays true, and the
    dive is re-selected every hour forever. That is the same never-goes-false
    shape that blocked scheduling stage 14 before 2026-07-17.

    These `LIKE` patterns approximate `taxonomy.is_measurable`, which is the
    definition of record. `test_dive_pipeline_status_view.py` runs the SQL
    over `taxonomy.MEASURABILITY_CORPUS` and asserts the two agree.
    """
    return (  # pylint: disable=no-member
        or_(
            SpeciesLabel.content_of_image.like(taxonomy.REAL_FISH_LIKE),
            _is_fish_model_condition(),
        ),
    )


def _is_fish_model_condition():
    """A rigid known-length target (fish model or the ruler).

    These carry no grouping labels and thus no LABEL_STUDIO cluster, so the
    stage-14 cohort waives the cluster requirement for them: identity is the
    target name, and length uses only laser/head-tail/calibration.
    Mirrors `views._IS_FISH_MODEL_SQL`.
    """
    # pylint: disable=no-member
    return or_(
        and_(
            SpeciesLabel.content_of_image.like(taxonomy.FISH_MODEL_LIKE),
            # Excludes the empty leaf "Fish Model," — see
            # `taxonomy.rigid_target_sql` for why that one matters.
            func.trim(SpeciesLabel.content_of_image) != taxonomy.FISH_MODEL_PREFIX,
        ),
        SpeciesLabel.content_of_image == taxonomy.RULER_CONTENT,
    )


def _valid_headtail_conditions():
    """The repo-wide definition of a *valid* head/tail label.

    Both keypoints fully placed. Only stage 14 needs this — every other
    stage only cares that a HeadTailLabel row exists at all — but it
    mirrors `_valid_laser_conditions` so the pair reads together.

    Mirrors `views._VALID_HEADTAIL_SQL`.
    """
    return (
        HeadTailLabel.completed == True,
        HeadTailLabel.superseded == False,
        HeadTailLabel.head_x != None,
        HeadTailLabel.head_y != None,
        HeadTailLabel.tail_x != None,
        HeadTailLabel.tail_y != None,
    )

def _resolved_laser_extrinsics_id():
    """The `LaserExtrinsics.id` a dive would actually be processed with.

    Own row wins, else the one borrowed through `Dive.calibration_dive_id` —
    the same own-then-link order `get_laser_extrinsics_for_dive` applies, so a
    stored provenance id can be compared against what the data-worker will
    really use. `uq_laserextrinsics_dive_id` guarantees at most one row per
    dive, so neither branch needs a tie-break.
    """
    # `.correlate(Dive)` is load-bearing, not decoration. This subquery is
    # used inside a NOT EXISTS several levels down, and SQLAlchemy only
    # auto-correlates against the *immediately* enclosing SELECT — so without
    # it the compiler emits `FROM laserextrinsics, dive`, an uncorrelated
    # cross join returning one row per extrinsics row in the table.
    #
    # Postgres rejects that with CardinalityViolationError ("more than one row
    # returned by a subquery used as an expression"), which took both cohort
    # selectors down in prod on 2026-08-20: every hourly poll 500'd, stage 14
    # stamped no provenance at all, and the laser-depth stage drained one dive
    # and then stopped. SQLite returns the first row instead of raising, so
    # the unit suite could not see it — `test_resolved_extrinsics_subquery_is_correlated`
    # now asserts the emitted SQL directly, and a sibling test seeds a second
    # extrinsics row so the uncorrelated form picks the wrong dive's
    # calibration even on SQLite.
    own = (
        select(LaserExtrinsics.id)
        .where(LaserExtrinsics.dive_id == Dive.id)
        .correlate(Dive)
        .scalar_subquery()
    )
    borrowed = (
        select(LaserExtrinsics.id)
        .where(LaserExtrinsics.dive_id == Dive.calibration_dive_id)
        .correlate(Dive)
        .scalar_subquery()
    )
    return func.coalesce(own, borrowed)


# Stage-13 cohort threshold; matches the data-worker calibration
# activity's `MIN_LASER_POINTS = 2` precondition. Selecting a dive with
# fewer than two completed slate labels would dispatch a child that
# raises and re-fires every hour.
MIN_COMPLETED_SLATE_LABELS = 2

# Stage-9 species_label.content_of_image marker (re-exported from the
# shared taxonomy vocabulary so the view and the worker read the same one).
SLATE_CONTENT_MARKER = taxonomy.SLATE_CONTENT_MARKER


# Cohort selectors used by the api-workflow-worker hourly schedules.
# Each returns the lowest-id HIGH-priority dive whose pipeline state
# matches the per-stage cohort, or None when the cohort is empty. The
# predicate moves to a single SELECT … LIMIT 1 — the pre-existing
# client-side N+1 loop in the worker activities was timing out
# schedule_to_close on backlogs of a few hundred dives.
#
# These routes must be declared before `get_dive` because FastAPI
# matches declaration order: `/dives/select-next/...` would otherwise
# try to coerce "select-next" into the `{dive_id}: int` path param and
# 422.


@app.get("/api/v1/dives/select-next/laser-preprocessing/")
async def select_next_for_laser_preprocessing(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 0.1: HIGH-priority + at least one image without a
    non-sentinel LaserLabel row (in any real project).

    "Non-sentinel" means `label_studio_project_id IS NOT NULL` —
    NULL-project rows are legacy sentinels (prod has ~2000 of them, one
    per HIGH-priority canonical image, source unclear but predates the
    Create-on-populate flow). The convention is established already:
    every discovery endpoint in `label_controller.py` filters
    `project_id != None` for the same reason.

    Predicate is "non-sentinel row exists?" not "row completed?" so a
    dive drops out of the cohort the moment populate seeds even-
    incomplete rows for every image. Without that, every preprocessed
    dive would stay in the cohort until labelers finished it — an
    hourly firing of stage 0.1 would re-stage raw `.ORF`s from NAS,
    re-rectify, and re-archive (the data-worker child workflow's
    ALLOW_DUPLICATE_FAILED_ONLY policy makes that a no-op, but the NAS
    staging activity runs unconditionally on every parent firing).
    """
    has_image_without_real_laser_label = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(
            ~select(LaserLabel.id)
            .where(LaserLabel.image_id == Image.id)
            .where(LaserLabel.label_studio_project_id != None)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_image_without_real_laser_label)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/laser-prediction/")
async def select_next_for_laser_prediction(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Model-assisted laser labeling: HIGH-priority + at least one image
    with no `LaserPrediction` row and no *completed* `LaserLabel` row.

    An image needs a prediction only if it has neither been predicted nor
    *labeled by a human* yet — so a dive drops out of the cohort once every
    image is predicted (one-shot per image; re-prediction is a manual
    affair), and images a human already labeled are never predicted over.

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


@app.get("/api/v1/dives/select-next/dive-frame-clustering/")
async def select_next_for_dive_frame_clustering(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 1: HIGH-priority + has at least one image carrying a
    *valid* LaserLabel (completed=True, superseded=False, x/y both
    populated) AND has zero PREDICTION DiveFrameCluster rows.

    Cascades from valid lasers like the headtail/species pipelines
    do — clustering is the prerequisite for stage 2 species
    preprocessing, so it should fire as soon as labelers + the
    validator sign off on lasers. The "no PREDICTION cluster"
    half is the one-shot gate: clustering is per-dive and
    deterministic on the timestamp set, so once it has run we
    don't need to re-run.
    """
    has_valid_laser_image = (
        select(LaserLabel.id)
        .join(Image, Image.id == LaserLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(*_valid_laser_conditions())
        .exists()
    )
    has_prediction_cluster = (
        select(DiveFrameCluster.id)
        .where(DiveFrameCluster.dive_id == Dive.id)
        .where(DiveFrameCluster.data_source == DataSource.PREDICTION)
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_valid_laser_image)
        .where(~has_prediction_cluster)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/species-preprocessing/")
async def select_next_for_species_preprocessing(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 2: HIGH-priority + has PREDICTION cluster + at least one
    image carrying a *valid* LaserLabel (completed, not superseded,
    both x/y populated) whose image carries no non-sentinel
    SpeciesLabel row.

    Cohort flipped from species-only ("any image without species
    label") → laser-cascade on 2026-05-05 so species labeling fires
    in parallel with head/tail (5.1) as soon as laser labelers + the
    validator sign off, while still waiting on stage-1 clustering to
    land PREDICTION clusters that the data-worker fan-out needs for
    the cluster-overlay context.

    "Valid laser" matches the predicate already used by
    `perform_laser_calibration_activity`,
    `validate_laser_labels_for_dive_activity._positive_xy`, and the
    headtail cohort.

    "Non-sentinel" species = `project_id IS NOT NULL` AND not
    superseded. See the laser cohort docstring for the sentinel
    rationale (sentinels predate the new flow and every other discovery
    query already filters them out).

    The superseded half was added 2026-07-21: a dead-lettered row is not
    evidence the work is done, and treating it as such deadlocked the
    stage against `needing-species-population` (which is superseded-
    aware). 1,826 species and 1,761 headtail images were stranded — their
    JPEGs never regenerated, so populate deferred them forever and no
    per-dive species project could publish.

    The PREDICTION-cluster gate is checked on the SAME image as the laser
    gate, not dive-wide (fixed 2026-07-22). `resolve_species_preprocess_inputs_activity`
    only dispatches per-image work for a qualifying image that is *in* a
    PREDICTION cluster (it needs the cluster for the "image i of N" overlay).
    The selector used to check "dive has some cluster" and "dive has some
    qualifying image" independently, so a dive whose one qualifying image
    was NOT clustered (a laser validated after stage-1 clustering, which is
    one-shot per dive) got selected while the resolver returned zero. Since
    the parent drains one dive per hour ordered by id, such a dive sat at the
    front forever, resolving to nothing and starving every productive dive
    behind it — dives 59 and 439 did exactly this, blocking 60/61/66/76/…
    """
    has_valid_laser_image_in_cluster_without_real_species = (
        select(LaserLabel.id)
        .join(Image, Image.id == LaserLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(*_valid_laser_conditions())
        .where(
            ~select(SpeciesLabel.id)
            .where(SpeciesLabel.image_id == Image.id)
            .where(SpeciesLabel.label_studio_project_id != None)
            # A superseded row is a dead letter, not evidence the work is
            # done. Without this, an image whose only species row was
            # dead-lettered (retired old-LS project) never gets its stage-2
            # JPEG regenerated, so populate defers it forever and the
            # per-dive project never publishes.
            .where(SpeciesLabel.superseded == False)
            .exists()
        )
        # The qualifying image must itself be in a PREDICTION cluster — this
        # is what the resolver requires, so checking it here keeps the two in
        # step. Subsumes the old dive-wide "has any PREDICTION cluster" gate.
        .where(
            select(DiveFrameClusterImageMapping.image_id)
            .join(
                DiveFrameCluster,
                DiveFrameCluster.id
                == DiveFrameClusterImageMapping.dive_frame_cluster_id,
            )
            .where(DiveFrameClusterImageMapping.image_id == Image.id)
            .where(DiveFrameCluster.data_source == DataSource.PREDICTION)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_valid_laser_image_in_cluster_without_real_species)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/needing-species-population/")
async def select_dives_needing_species_population(
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Dives that need species LS tasks (re)populated onto a live project.

    Cohort: HIGH priority + at least one image carrying a *valid*
    LaserLabel (completed, not superseded, both x/y populated) that has
    no *non-superseded* SpeciesLabel row with a `project_id` — i.e. no
    live species task. Superseded rows are dead-lettered and don't
    count, which is exactly what lets a dive whose old-project rows were
    superseded (e.g. after the hosted-LS migration) re-enter the cohort.

    `select-next/species-preprocessing` is superseded-aware too, as of
    2026-07-21 — it previously ignored supersede, which permanently
    excluded migrated dives from stage 2 while this endpoint kept
    re-selecting them for populate. The two disagreeing was a deadlock:
    populate deferred every image whose JPEG preprocess would never
    regenerate, so `deferred > 0` and the project never published.
    This endpoint still drops the PREDICTION-cluster gate — populate
    only needs the species JPEGs to exist, which the populate activity
    gates per-image against Garage.
    The activity is idempotent + JPEG-gated, so this coarse candidate
    set is safe to over-select.

    Returns every matching dive id; the scheduled populate parent fans
    out one `PopulateSpeciesLabelStudioProjectWorkflow` child per dive.
    """
    has_valid_laser_image_without_live_species = (
        select(LaserLabel.id)
        .join(Image, Image.id == LaserLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(LaserLabel.completed == True)
        .where(LaserLabel.superseded == False)
        .where(LaserLabel.x != None)
        .where(LaserLabel.y != None)
        .where(
            ~select(SpeciesLabel.id)
            .where(SpeciesLabel.image_id == Image.id)
            .where(SpeciesLabel.label_studio_project_id != None)
            .where(SpeciesLabel.superseded == False)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_valid_laser_image_without_live_species)
        .order_by(Dive.id)
    )
    return list((await session.exec(query)).all())


@app.get("/api/v1/dives/needing-laser-population/")
async def select_dives_needing_laser_population(
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Dives that need model-assisted laser LS tasks (re)populated.

    Cohort: HIGH priority + at least one image that has a `LaserPrediction`
    (the detector has run for it) and no *completed* `LaserLabel`. The
    prediction gate is what makes this the decoupled populate cohort rather
    than the preprocess one: laser populate seeds non-sentinel `LaserLabel`
    rows, and the predict cohort excludes any image with a `LaserLabel`, so
    populating an un-predicted image would starve it of a prediction forever.
    Gating on "prediction exists" guarantees the detector ran first.

    "No completed label" (not "no row") keeps a dive in the cohort until its
    labelers finish, so the idempotent populate self-heals hourly. Returns
    every match; the scheduled populate parent fans out one
    `PopulateLaserLabelStudioProjectWorkflow` child per dive.
    """
    has_predicted_incomplete_image = (
        select(Image.id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(
            select(LaserPrediction.id)
            .where(LaserPrediction.image_id == Image.id)
            .exists()
        )
        .where(
            ~select(LaserLabel.id)
            .where(LaserLabel.image_id == Image.id)
            .where(LaserLabel.completed == True)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_predicted_incomplete_image)
        .order_by(Dive.id)
    )
    return list((await session.exec(query)).all())


@app.get("/api/v1/dives/select-next/headtail-preprocessing/")
async def select_next_for_headtail_preprocessing(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 5.1: HIGH-priority + has at least one image carrying a
    *valid* LaserLabel (completed, not superseded, both x/y populated)
    whose image carries no non-sentinel HeadTailLabel row.

    Cascade source flipped from species top-3 → valid laser labels on
    2026-05-04 so head/tail labeling fans out as soon as laser
    labelers (and the validator) sign off, without waiting for the
    species pass. "Valid" matches the predicate already used by
    `perform_laser_calibration_activity` and
    `validate_laser_labels_for_dive_activity._positive_xy`:
    null x/y are sentinel/no-laser rows, superseded comes from
    validation, completed comes from the labeler.

    "Non-sentinel" headtail = `project_id IS NOT NULL`. See the laser
    cohort docstring for the rationale.
    """
    has_valid_laser_image_without_real_headtail = (
        select(LaserLabel.id)
        .join(Image, Image.id == LaserLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(*_valid_laser_conditions())
        .where(
            ~select(HeadTailLabel.id)
            .where(HeadTailLabel.image_id == Image.id)
            .where(HeadTailLabel.label_studio_project_id != None)
            # Dead letters don't count as done — see the species cohort.
            .where(HeadTailLabel.superseded == False)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_valid_laser_image_without_real_headtail)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/slate-preprocessing/")
async def select_next_for_slate_preprocessing(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 9: HIGH-priority + dive_slate_id set + has at least one
    SpeciesLabel.content_of_image='Slate, Laser on slate' whose image
    carries no non-sentinel DiveSlateLabel row.

    "Non-sentinel" = `project_id IS NOT NULL`. See the laser cohort
    docstring for the rationale.
    """
    has_slate_marked_image_without_real_dive_slate_label = (
        select(SpeciesLabel.id)
        .join(Image, Image.id == SpeciesLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(SpeciesLabel.content_of_image == SLATE_CONTENT_MARKER)
        .where(
            ~select(DiveSlateLabel.id)
            .where(DiveSlateLabel.image_id == Image.id)
            .where(DiveSlateLabel.label_studio_project_id != None)
            # Dead letters don't count as done — see the species cohort.
            # No slate rows are superseded today; this keeps the three
            # preprocess gates spelled identically so the next supersede
            # pass can't strand slate images the way it stranded species.
            .where(DiveSlateLabel.superseded == False)
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(Dive.dive_slate_id != None)
        .where(has_slate_marked_image_without_real_dive_slate_label)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/laser-calibration/")
async def select_next_for_laser_calibration(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 13: HIGH-priority + dive_slate_id set + no LaserExtrinsics +
    at least MIN_COMPLETED_SLATE_LABELS completed DiveSlateLabel rows."""
    completed_slate_label_count = (
        select(func.count(DiveSlateLabel.id))  # pylint: disable=not-callable
        .join(Image, Image.id == DiveSlateLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(DiveSlateLabel.completed == True)
        # A dead-lettered slate label doesn't count toward the calibration
        # readiness gate — same validity convention laser calibration uses.
        .where(DiveSlateLabel.superseded == False)
        .scalar_subquery()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(Dive.dive_slate_id != None)
        .where(
            ~select(LaserExtrinsics.id)
            .where(LaserExtrinsics.dive_id == Dive.id)
            .exists()
        )
        .where(completed_slate_label_count >= MIN_COMPLETED_SLATE_LABELS)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/measure-fish/")
async def select_next_for_measure_fish(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Stage 14: HIGH-priority + has LaserExtrinsics + has at least one
    *measurable* image with no Measurement.

    "Measurable" mirrors what measure_fish_activity attempts, and this
    predicate mirrors `dive_pipeline_status.measured` — keep the two in
    step.

    Previously keyed on "has a LABEL_STUDIO cluster with fish_id IS
    NULL", which never went false: a cluster is only bound to a fish
    through a measurable image, so any cluster without one kept the dive
    in the cohort permanently (prod dive 466 carried 1632 such clusters
    against 24 measurable images). Combined with the old non-idempotent
    write, a scheduled stage 14 would have re-measured the same dives
    every hour. Both halves are fixed now — measurement upserts on
    (image_id, fish_id) and the activity skips already-measured images —
    so this cohort drains and the workflow is safe to schedule.
    """
    # Own calibration, or a sibling dive's via `calibration_dive_id`.
    # When the link is NULL the borrowed EXISTS never matches (no
    # LaserExtrinsics row has dive_id = NULL), so it reduces to the own
    # check. Mirrors `dive_pipeline_status.calibrated`.
    has_laser_extrinsics = or_(
        select(LaserExtrinsics.id)
        .where(LaserExtrinsics.dive_id == Dive.id)
        .exists(),
        select(LaserExtrinsics.id)
        .where(LaserExtrinsics.dive_id == Dive.calibration_dive_id)
        .exists(),
    )
    valid_laser = (
        select(LaserLabel.id)
        .where(LaserLabel.image_id == Image.id)
        .where(*_valid_laser_conditions())
        .exists()
    )
    valid_headtail = (
        select(HeadTailLabel.id)
        .where(HeadTailLabel.image_id == Image.id)
        .where(*_valid_headtail_conditions())
        .exists()
    )
    in_label_studio_cluster = (
        select(DiveFrameClusterImageMapping.image_id)
        .join(
            DiveFrameCluster,
            DiveFrameCluster.id == DiveFrameClusterImageMapping.dive_frame_cluster_id,
        )
        .where(DiveFrameClusterImageMapping.image_id == Image.id)
        .where(DiveFrameCluster.data_source == DataSource.LABEL_STUDIO)
        .exists()
    )
    # Not merely "has a Measurement": has one computed with the calibration
    # this dive would be measured with *today*. A length is a function of the
    # extrinsics behind its depth, so a recalibration invalidates it, and
    # rows written before `Measurement.laser_extrinsics_id` existed carry
    # NULL and never match — which is how the pre-provenance backlog
    # re-enters the cohort, gets recomputed once, and drains. Mirrored by
    # `dive_pipeline_status.measured`.
    is_measured = (
        select(Measurement.id)
        .where(Measurement.image_id == Image.id)
        .where(Measurement.laser_extrinsics_id == _resolved_laser_extrinsics_id())
        .exists()
    )
    has_unmeasured_measurable_image = (
        select(SpeciesLabel.id)
        .join(Image, Image.id == SpeciesLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(SpeciesLabel.top_three_photos_of_group == True)
        .where(*_measurable_species_conditions())
        .where(valid_laser)
        .where(valid_headtail)
        # Real fish need a LABEL_STUDIO cluster; fish models carry none and
        # need none (see `_is_fish_model_condition`).
        .where(or_(in_label_studio_cluster, _is_fish_model_condition()))
        .where(~is_measured)
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_laser_extrinsics)
        .where(has_unmeasured_measurable_image)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


@app.get("/api/v1/dives/select-next/laser-depth/")
async def select_next_for_laser_depth(
    session: AsyncSession = Depends(get_async_session),
) -> int | None:
    """Laser depth: HIGH-priority + resolvable calibration + at least one
    canonical image whose valid laser label has no *current* `LaserDepth`.

    "Current" is the point. Depth is a function of the laser label and the
    calibration, so the stored row names both, and an image counts as needing
    work when either differs from what would be used today — no row at all, a
    replaced label, or a recalibration. That is what lets a recompute be an
    ordinary drainable cohort instead of a hand-run backfill: the 2026-08-11
    panel-offset recalibration would re-enter its dives here automatically.

    Unlike stage 14 this does not require head/tail labels, clusters, or a
    measurable species — the dot's distance is knowable for any frame whose
    laser was validated, which is the whole reason it is stored per image
    rather than hung off `Measurement`.
    """
    return (await session.exec(_laser_depth_cohort_query())).first()


def _laser_depth_cohort_query():
    """The laser-depth cohort as a query, separate from executing it.

    Split out so `test_resolved_extrinsics_subquery_is_correlated` can compile
    the *real* query and assert its shape. That guard has to see the actual
    nesting — the correlation bug it exists to catch only appears when the
    scalar subquery sits several levels inside a NOT EXISTS, so a synthetic
    one-level query rebuilt in the test would pass while the shipped one was
    broken.
    """
    has_laser_extrinsics = or_(
        select(LaserExtrinsics.id).where(LaserExtrinsics.dive_id == Dive.id).exists(),
        select(LaserExtrinsics.id)
        .where(LaserExtrinsics.dive_id == Dive.calibration_dive_id)
        .exists(),
    )
    depth_is_current = (
        select(LaserDepth.id)
        .where(LaserDepth.image_id == Image.id)
        .where(LaserDepth.laser_label_id == LaserLabel.id)
        .where(LaserDepth.laser_extrinsics_id == _resolved_laser_extrinsics_id())
        .exists()
    )
    has_image_needing_depth = (
        select(LaserLabel.id)
        .join(Image, Image.id == LaserLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(Image.is_canonical == True)
        .where(*_valid_laser_conditions())
        .where(~depth_is_current)
        .exists()
    )
    return (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_laser_extrinsics)
        .where(has_image_needing_depth)
        .order_by(Dive.id)
        .limit(1)
    )

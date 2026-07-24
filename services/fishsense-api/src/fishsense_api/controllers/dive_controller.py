# pylint: disable=C0121
"""Dive Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import func, or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.data_source import DataSource
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_frame_cluster import (
    DiveFrameCluster,
    DiveFrameClusterImageMapping,
)
from fishsense_api.models.dive_slate import DiveSlate
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.measurement import Measurement
from fishsense_api.models.priority import Priority
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)


def _valid_laser_conditions():
    """The repo-wide definition of a *valid* laser label.

    The labeler placed a point, the validator signed off, and
    `ValidateLaserLabelsForDiveWorkflow`'s RANSAC fit hasn't superseded
    it. Stages 1, 2, 5.1 and 14 all cascade from this gate, so it's
    spelled once. Splat into a select: `.where(*_valid_laser_conditions())`.

    Mirrors `views._VALID_LASER_SQL` — the view and these selectors are
    two representations of the same predicate and must stay in step.
    """
    return (
        LaserLabel.completed == True,
        LaserLabel.superseded == False,
        LaserLabel.x != None,
        LaserLabel.y != None,
    )


def _measurable_species_conditions():
    """A species row stage 14 can actually turn into a Measurement.

    `measure_fish_activity._parse_species_names` reads the species name from
    the LAST ", "-separated chunk of `content_of_image` and requires the
    `Common Name (Scientific name)` shape, returning None otherwise — the
    activity then skips the image rather than writing a malformed Species
    row. Only the `Fish` taxonomy branch carries that shape:

        "Fish, Hogfish (Lachnolaimus maximus)"  -> measurable
        "Fish Model, Weasly Fish"               -> skipped (no parens)
        "Calibration Targets, Ruler"            -> skipped (no parens)

    Without this, the cohort and the activity disagree in a way that cannot
    resolve: the selector keeps offering an image the activity always skips,
    so no Measurement is written, `~is_measured` stays true, and the dive is
    re-selected every hour forever. That is the same never-goes-false shape
    that blocked scheduling stage 14 before 2026-07-17.

    Mirrors `views._MEASURABLE_SPECIES_SQL` — keep the two in step.
    """
    return (SpeciesLabel.content_of_image.like("%(%)"),)  # pylint: disable=no-member


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

# Stage-13 cohort threshold; matches the data-worker calibration
# activity's `MIN_LASER_POINTS = 2` precondition. Selecting a dive with
# fewer than two completed slate labels would dispatch a child that
# raises and re-fires every hour.
MIN_COMPLETED_SLATE_LABELS = 2

# Stage-9 species_label.content_of_image marker.
SLATE_CONTENT_MARKER = "Slate, Laser on slate"


@app.get("/api/v1/dives/")
async def get_dives(session: AsyncSession = Depends(get_async_session)) -> List[Dive]:
    """Retrieve all dives."""
    logger.debug("Retrieving all dives")
    query = select(Dive)

    return (await session.exec(query)).all()


@app.get("/api/v1/canonical/dives/")
async def get_canonical_dives(
    session: AsyncSession = Depends(get_async_session),
) -> List[Dive]:
    """Retrieve all canonical dives."""
    logger.debug("Retrieving all canonical dives")
    query = (
        select(Dive)
        .distinct(Dive.id)
        .join_from(Dive, Image, Dive.id == Image.dive_id)
        .where(Image.is_canonical == True)
    )

    result = await session.exec(query)
    return result.all()


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
    is_measured = (
        select(Measurement.id).where(Measurement.image_id == Image.id).exists()
    )
    has_unmeasured_measurable_image = (
        select(SpeciesLabel.id)
        .join(Image, Image.id == SpeciesLabel.image_id)
        .where(Image.dive_id == Dive.id)
        .where(SpeciesLabel.top_three_photos_of_group == True)
        .where(*_measurable_species_conditions())
        .where(valid_laser)
        .where(valid_headtail)
        .where(in_label_studio_cluster)
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


@app.get("/api/v1/dives/{dive_id}")
async def get_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> Dive | None:
    """Retrieve a dive by its ID."""
    logger.debug("Retrieving dive with id=%d", dive_id)
    query = select(Dive).where(Dive.id == dive_id)

    dive = (await session.exec(query)).first()
    if dive is None:
        logger.warning("Dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Dive not found")
    return dive


def _latest_extrinsics_query(dive_id: int):
    """The most-recently-created `LaserExtrinsics` row for a dive."""
    return (
        select(LaserExtrinsics)
        .where(LaserExtrinsics.dive_id == dive_id)
        .where(
            LaserExtrinsics.created_at
            == select(LaserExtrinsics.created_at)
            .where(LaserExtrinsics.dive_id == dive_id)
            .order_by(LaserExtrinsics.created_at.desc())  # pylint: disable=no-member
            .limit(1)
            .scalar_subquery()
        )
    )


@app.get("/api/v1/dives/{dive_id}/laser-extrinsics/")
async def get_laser_extrinsics_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserExtrinsics | None:
    """Retrieve the laser extrinsics that apply to a dive.

    A dive's *own* calibration wins; if it has none but is linked to a
    calibration-source dive (`Dive.calibration_dive_id`), the source
    dive's extrinsics are returned instead. This lets a fish-only dive
    with no slate frames borrow the calibration of a sibling slate dive
    shot with the same camera+laser rig. The `laser_position` /
    `laser_axis` are all stage 14 consumes, so the returned row's
    `dive_id` (the source dive) is inconsequential to callers.
    """
    logger.debug("Retrieving laser extrinsics for dive with id=%d", dive_id)

    laser_extrinsics = (
        await session.exec(_latest_extrinsics_query(dive_id))
    ).first()

    if laser_extrinsics is None:
        dive = await session.get(Dive, dive_id)
        if dive is not None and dive.calibration_dive_id is not None:
            logger.debug(
                "dive id=%d has no own extrinsics; borrowing from "
                "calibration_dive_id=%d",
                dive_id,
                dive.calibration_dive_id,
            )
            laser_extrinsics = (
                await session.exec(
                    _latest_extrinsics_query(dive.calibration_dive_id)
                )
            ).first()

    if laser_extrinsics is None:
        logger.warning("Laser extrinsics for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Laser extrinsics not found")
    return laser_extrinsics


@app.put("/api/v1/dives/{dive_id}/laser-extrinsics/", status_code=201)
async def put_laser_extrinsics_for_dive(
    dive_id: int,
    extrinsics: LaserExtrinsics,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update laser extrinsics for a given dive ID."""
    logger.debug("Creating or updating laser extrinsics for dive with id=%d", dive_id)
    extrinsics = LaserExtrinsics.model_validate(jsonable_encoder(extrinsics))
    extrinsics.dive_id = dive_id

    extrinsics = await session.merge(extrinsics)
    await session.flush()

    extrinsics_id = extrinsics.id

    return extrinsics_id


@app.put("/api/v1/dives/{dive_id}/calibration-source/{source_dive_id}")
async def set_dive_calibration_source(
    dive_id: int,
    source_dive_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Link `dive_id` to borrow `source_dive_id`'s laser calibration.

    For a fish-only dive with no slate frames of its own: point it at a
    sibling slate/calibration dive shot with the same camera+laser rig.
    Laser-extrinsics resolution and the `calibrated` gate then fall back
    to the source dive when this dive has no calibration of its own.

    Returns the linked dive's id. 404 if either dive is missing; 400 on a
    self-link (a dive is never its own calibration source).
    """
    logger.debug(
        "Linking dive id=%d to calibration source dive id=%d",
        dive_id,
        source_dive_id,
    )
    if dive_id == source_dive_id:
        raise HTTPException(
            status_code=400,
            detail="A dive cannot be its own calibration source",
        )

    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    source = await session.get(Dive, source_dive_id)
    if source is None:
        raise HTTPException(
            status_code=404, detail="Calibration source dive not found"
        )

    dive.calibration_dive_id = source_dive_id
    session.add(dive)
    await session.flush()

    return dive_id


@app.delete("/api/v1/dives/{dive_id}/calibration-source/", status_code=204)
async def clear_dive_calibration_source(
    dive_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Unlink `dive_id` from any borrowed calibration source (idempotent).

    404 only if the dive itself is missing; clearing an already-null link
    is a no-op.
    """
    logger.debug("Clearing calibration source link for dive id=%d", dive_id)
    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    dive.calibration_dive_id = None
    session.add(dive)
    await session.flush()


@app.put("/api/v1/dives/{dive_id}/dive-slate/{dive_slate_id}")
async def set_dive_slate(
    dive_id: int,
    dive_slate_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Set which `DiveSlate` template a dive was shot with.

    Identifies the physical slate (H-Slate / V-Slate N / Tic-Tac-Toe N),
    which stages 9/12/13 need before a dive can be slate-labeled and
    calibrated. Populated by the species-label sync from the labeler's
    slate-type choice; also settable by an operator.

    Returns the dive id. 404 if the dive or the DiveSlate template is
    missing.
    """
    logger.debug(
        "Setting dive id=%d dive_slate_id=%d", dive_id, dive_slate_id
    )
    dive = await session.get(Dive, dive_id)
    if dive is None:
        raise HTTPException(status_code=404, detail="Dive not found")

    slate = await session.get(DiveSlate, dive_slate_id)
    if slate is None:
        raise HTTPException(status_code=404, detail="DiveSlate template not found")

    dive.dive_slate_id = dive_slate_id
    session.add(dive)
    await session.flush()

    return dive_id

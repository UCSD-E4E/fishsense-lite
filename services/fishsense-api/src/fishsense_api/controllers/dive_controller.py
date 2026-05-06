# pylint: disable=C0121
"""Dive Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import func
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.data_source import DataSource
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.priority import Priority
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)

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
        .where(LaserLabel.completed == True)
        .where(LaserLabel.superseded == False)
        .where(LaserLabel.x != None)
        .where(LaserLabel.y != None)
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

    "Non-sentinel" species = `project_id IS NOT NULL`. See the laser
    cohort docstring for the rationale (sentinels predate the new
    flow and every other discovery query already filters them out).
    """
    has_prediction_cluster = (
        select(DiveFrameCluster.id)
        .where(DiveFrameCluster.dive_id == Dive.id)
        .where(DiveFrameCluster.data_source == DataSource.PREDICTION)
        .exists()
    )
    has_valid_laser_image_without_real_species = (
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
            .exists()
        )
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_prediction_cluster)
        .where(has_valid_laser_image_without_real_species)
        .order_by(Dive.id)
        .limit(1)
    )
    return (await session.exec(query)).first()


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
        .where(LaserLabel.completed == True)
        .where(LaserLabel.superseded == False)
        .where(LaserLabel.x != None)
        .where(LaserLabel.y != None)
        .where(
            ~select(HeadTailLabel.id)
            .where(HeadTailLabel.image_id == Image.id)
            .where(HeadTailLabel.label_studio_project_id != None)
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
    LABEL_STUDIO cluster with fish_id IS NULL.

    First-run gate, not a strict idempotency gate — measure_fish_activity
    is non-idempotent (POST measurement, no per-image filter), so a
    partially-failed run will re-fire and duplicate measurements on
    already-bound clusters. Caller (parent workflow) is operator-
    triggered, not scheduled."""
    has_laser_extrinsics = (
        select(LaserExtrinsics.id)
        .where(LaserExtrinsics.dive_id == Dive.id)
        .exists()
    )
    has_unbound_label_studio_cluster = (
        select(DiveFrameCluster.id)
        .where(DiveFrameCluster.dive_id == Dive.id)
        .where(DiveFrameCluster.data_source == DataSource.LABEL_STUDIO)
        .where(DiveFrameCluster.fish_id == None)
        .exists()
    )
    query = (
        select(Dive.id)
        .where(Dive.priority == Priority.HIGH)
        .where(has_laser_extrinsics)
        .where(has_unbound_label_studio_cluster)
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


@app.get("/api/v1/dives/{dive_id}/laser-extrinsics/")
async def get_laser_extrinsics_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserExtrinsics | None:
    """Retrieve all laser extrinsics for a given dive ID."""
    logger.debug("Retrieving laser extrinsics for dive with id=%d", dive_id)
    # With max date filter
    query = (
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

    laser_extrinsics = (await session.exec(query)).first()
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

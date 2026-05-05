# pylint: disable=C0121
"""Label Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy import alias
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.label_studio_sync_cursor import LabelStudioSyncCursor
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/labels/dive-slate/label-studio-project-ids")
async def get_dive_slate_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one dive-slate label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    NOTE: must precede the `/api/v1/labels/dive-slate/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with dive-slate labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = select(DiveSlateLabel.label_studio_project_id).where(
        DiveSlateLabel.label_studio_project_id != None
    )
    if incomplete:
        query = query.where(
            (DiveSlateLabel.completed == False)
            | (DiveSlateLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/dive-slate/{image_id}")
async def get_dive_slate_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> DiveSlateLabel | None:
    """Retrieve slate label for a given image ID."""
    logger.debug("Retrieving dive slate label for image with id=%d", image_id)

    query = select(DiveSlateLabel).where(DiveSlateLabel.image_id == image_id)

    return (await session.exec(query)).first()


@app.get("/api/v1/dives/{dive_id}/labels/dive-slate")
async def get_dive_slate_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[DiveSlateLabel]:
    """Retrieve all slate labels for a given dive ID."""
    logger.debug("Retrieving dive slate labels for dive with id=%d", dive_id)
    query = (
        select(DiveSlateLabel)
        .join_from(DiveSlateLabel, Image, DiveSlateLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Dive slate labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.put("/api/v1/labels/dive-slate/{image_id}", status_code=201)
async def put_dive_slate_label(
    image_id: int,
    label: DiveSlateLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update slate label for a given image ID."""
    logger.debug("Creating or updating dive slate label for image with id=%d", image_id)
    label = DiveSlateLabel.model_validate(jsonable_encoder(label))
    label.image_id = image_id

    label = await session.merge(label)
    await session.flush()

    label_id = label.id

    return label_id


@app.get("/api/v1/labels/dive-slate/label-studio/{label_studio_id}")
async def get_dive_slate_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> DiveSlateLabel | None:
    """Retrieve a dive-slate label for a given Label Studio task ID."""
    logger.debug(
        "Retrieving dive-slate label for Label Studio id=%d", label_studio_id
    )
    query = select(DiveSlateLabel).where(
        DiveSlateLabel.label_studio_task_id == label_studio_id
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Dive-slate label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/headtail/label-studio-project-ids")
async def get_headtail_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one head-tail label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    Replaces a per-dive fan-out the api-workflow-worker used to do (one
    HTTP round trip per canonical dive) — that approach blew past the
    activity's 10-minute schedule_to_close timeout as the dataset grew.

    NOTE: must precede the `/api/v1/labels/headtail/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with head-tail labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = select(HeadTailLabel.label_studio_project_id).where(
        HeadTailLabel.label_studio_project_id != None
    )
    if incomplete:
        query = query.where(
            (HeadTailLabel.completed == False)
            | (HeadTailLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/headtail/{image_id}")
async def get_headtail_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> HeadTailLabel | None:
    """Retrieve a head-tail label for a given image ID."""
    logger.debug("Retrieving head-tail label for image with id=%d", image_id)

    query = (
        select(HeadTailLabel)
        .where(HeadTailLabel.image_id == image_id)
        .where(HeadTailLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Head-tail label for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/dives/{dive_id}/labels/headtail")
async def get_headtail_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[HeadTailLabel]:
    """Retrieve all head-tail labels for a given dive ID."""
    logger.debug("Retrieving head-tail labels for dive with id=%d", dive_id)
    query = (
        select(HeadTailLabel)
        .join_from(HeadTailLabel, Image, HeadTailLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
        .where(HeadTailLabel.superseded == False)
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Head-tail labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.put("/api/v1/labels/headtail/{image_id}", status_code=201)
async def put_headtail_label(
    image_id: int,
    label: HeadTailLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a head-tail label for a given image ID."""
    logger.debug("Creating or updating head-tail label for image with id=%d", image_id)
    label = HeadTailLabel.model_validate(jsonable_encoder(label))
    label.image_id = image_id

    label = await session.merge(label)
    await session.flush()

    label_id = label.id

    return label_id


@app.get("/api/v1/labels/headtail/label-studio/{label_studio_id}")
async def get_headtail_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> HeadTailLabel | None:
    """Retrieve a head-tail label for a given Label Studio ID."""
    logger.debug(
        "Retrieving head-tail label for Label Studio id=%d", label_studio_id
    )
    query = (
        select(HeadTailLabel)
        .where(HeadTailLabel.label_studio_task_id == label_studio_id)
        .where(HeadTailLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Head-tail label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/laser/label-studio-project-ids")
async def get_laser_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one laser label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    Replaces a per-dive fan-out the api-workflow-worker used to do (one
    HTTP round trip per canonical dive) — that approach blew past the
    activity's 10-minute schedule_to_close timeout as the dataset grew.

    NOTE: must precede the `/api/v1/labels/laser/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with laser labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = select(LaserLabel.label_studio_project_id).where(
        LaserLabel.label_studio_project_id != None
    )
    if incomplete:
        query = query.where(
            (LaserLabel.completed == False)
            | (LaserLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/laser/dives-with-complete-labeling")
async def get_dives_with_complete_laser_labeling(
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Dive IDs whose laser labeling is fully complete.

    A dive qualifies iff every non-superseded `LaserLabel` for one of
    its images has `completed=True` AND at least one such label exists.
    Dives with zero non-superseded laser labels (no labeling activity at
    all) are excluded — there's nothing to validate.

    Backs the laser label-validation pass that runs after each laser
    sync: validating against an in-progress dive's labels is wasted
    effort because the line fit changes as more positives arrive.

    Implemented as `(dive has at least one completed non-superseded
    laser label) AND NOT (dive has any incomplete non-superseded laser
    label)` — `NOT EXISTS` is portable across the prod Postgres and
    the in-memory sqlite the integration tests use; `bool_and` would
    work on Postgres only.

    NOTE: must precede the `/api/v1/labels/laser/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug("Retrieving dive IDs with complete laser labeling")
    incomplete = alias(LaserLabel.__table__, name="incomplete_laser_label")
    incomplete_image = alias(Image.__table__, name="incomplete_image")
    has_incomplete = (
        select(incomplete.c.id)
        .select_from(
            incomplete.join(
                incomplete_image, incomplete.c.image_id == incomplete_image.c.id
            )
        )
        .where(incomplete_image.c.dive_id == Image.dive_id)
        .where(incomplete.c.superseded == False)
        .where(
            (incomplete.c.completed == False)
            | (incomplete.c.completed.is_(None))  # pylint: disable=no-member
        )
        .exists()
    )
    query = (
        select(Image.dive_id)
        .join(LaserLabel, LaserLabel.image_id == Image.id)
        .where(LaserLabel.superseded == False)
        .where(LaserLabel.completed == True)
        .where(Image.dive_id != None)
        .where(~has_incomplete)
        .distinct()
    )
    return list((await session.exec(query)).all())


@app.get("/api/v1/labels/laser/{image_id}")
async def get_laser_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserLabel | None:
    """Retrieve a laser label for a given image ID."""
    logger.debug("Retrieving laser label for image with id=%d", image_id)

    query = (
        select(LaserLabel)
        .where(LaserLabel.image_id == image_id)
        .where(LaserLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Laser label for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/laser/label-studio/{label_studio_id}")
async def get_laser_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserLabel | None:
    """Retrieve a laser label for a given Label Studio ID."""
    logger.debug("Retrieving laser label for Label Studio id=%d", label_studio_id)

    query = (
        select(LaserLabel)
        .where(LaserLabel.label_studio_task_id == label_studio_id)
        .where(LaserLabel.superseded == False)
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Laser label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/dives/{dive_id}/labels/laser")
async def get_laser_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[LaserLabel]:
    """Retrieve all laser labels for a given dive ID."""
    logger.debug("Retrieving laser labels for dive with id=%d", dive_id)
    query = (
        select(LaserLabel)
        .join_from(LaserLabel, Image, LaserLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
        .where(LaserLabel.superseded == False)
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Laser labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.put("/api/v1/labels/laser/{image_id}", status_code=201)
async def put_laser_label(
    image_id: int,
    label: LaserLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a laser label for a given image ID."""
    logger.debug("Creating or updating laser label for image with id=%d", image_id)
    label = LaserLabel.model_validate(jsonable_encoder(label))
    label.image_id = image_id

    label = await session.merge(label)
    await session.flush()

    label_id = label.id

    return label_id


@app.get("/api/v1/dives/{dive_id}/labels/species")
async def get_species_labels_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[SpeciesLabel]:
    """Retrieve all species labels for a given dive ID."""
    logger.debug("Retrieving species labels for dive with id=%d", dive_id)
    query = (
        select(SpeciesLabel)
        .join_from(SpeciesLabel, Image, SpeciesLabel.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )

    labels = (await session.exec(query)).all()
    if not labels:
        logger.warning("Species labels for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Labels not found")
    return labels


@app.get("/api/v1/labels/species/label-studio-project-ids")
async def get_species_label_studio_project_ids(
    incomplete: bool = False,
    session: AsyncSession = Depends(get_async_session),
) -> List[int]:
    """Distinct Label Studio project IDs that have at least one species label.

    `incomplete=true` narrows to projects that have at least one label
    where `completed` is NULL or false. Backs the `apps/fishsense-lite-web/` SSR
    landing page, which surfaces only LS projects with outstanding
    labeling work.

    NOTE: must precede the `/api/v1/labels/species/{image_id}` route —
    Starlette's default path converter treats `{image_id}` as `[^/]+`,
    so registration order is what disambiguates the literal segment.
    """
    logger.debug(
        "Retrieving distinct Label Studio project IDs with species labels "
        "(incomplete=%s)",
        incomplete,
    )
    query = select(SpeciesLabel.label_studio_project_id).where(
        SpeciesLabel.label_studio_project_id != None
    )
    if incomplete:
        query = query.where(
            (SpeciesLabel.completed == False)
            | (SpeciesLabel.completed.is_(None))  # pylint: disable=no-member
        )
    return list((await session.exec(query.distinct())).all())


@app.get("/api/v1/labels/species/{image_id}")
async def get_species_label(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> SpeciesLabel | None:
    """Retrieve a species label for a given image ID."""
    logger.debug("Retrieving species label for image with id=%d", image_id)
    query = select(SpeciesLabel).where(SpeciesLabel.image_id == image_id)

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning("Species label for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.put("/api/v1/labels/species/{image_id}", status_code=201)
async def put_species_label(
    image_id: int,
    label: SpeciesLabel,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a species label for a given image ID."""
    logger.debug("Creating or updating species label for image with id=%d", image_id)
    label = SpeciesLabel.model_validate(jsonable_encoder(label))
    label.image_id = image_id

    label = await session.merge(label)
    await session.flush()

    label_id = label.id

    return label_id


@app.get("/api/v1/labels/species/label-studio/{label_studio_id}")
async def get_species_label_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> SpeciesLabel | None:
    """Retrieve a species label for a given Label Studio task ID."""
    logger.debug(
        "Retrieving species label for Label Studio id=%d", label_studio_id
    )
    query = select(SpeciesLabel).where(
        SpeciesLabel.label_studio_task_id == label_studio_id
    )

    label = (await session.exec(query)).first()
    if label is None:
        logger.warning(
            "Species label for Label Studio id=%d not found", label_studio_id
        )
        raise HTTPException(status_code=404, detail="Label not found")
    return label


@app.get("/api/v1/labels/sync-cursor/{kind}/{label_studio_project_id}")
async def get_label_studio_sync_cursor(
    kind: str,
    label_studio_project_id: int,
    session: AsyncSession = Depends(get_async_session),
) -> LabelStudioSyncCursor | None:
    """Retrieve the sync cursor for a given (kind, project) pair.

    Returns None when no cursor exists yet — the api-workflow-worker
    treats that as "first run, sync everything."
    """
    query = (
        select(LabelStudioSyncCursor)
        .where(LabelStudioSyncCursor.kind == kind)
        .where(LabelStudioSyncCursor.label_studio_project_id == label_studio_project_id)
    )
    return (await session.exec(query)).first()


@app.put("/api/v1/labels/sync-cursor/{kind}/{label_studio_project_id}", status_code=201)
async def put_label_studio_sync_cursor(
    kind: str,
    label_studio_project_id: int,
    cursor: LabelStudioSyncCursor,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Upsert the sync cursor for a given (kind, project) pair."""
    cursor = LabelStudioSyncCursor.model_validate(jsonable_encoder(cursor))
    cursor.kind = kind
    cursor.label_studio_project_id = label_studio_project_id

    if cursor.id is None:
        existing_query = (
            select(LabelStudioSyncCursor)
            .where(LabelStudioSyncCursor.kind == kind)
            .where(
                LabelStudioSyncCursor.label_studio_project_id == label_studio_project_id
            )
        )
        existing = (await session.exec(existing_query)).first()
        if existing is not None:
            cursor.id = existing.id

    cursor = await session.merge(cursor)
    await session.flush()

    return cursor.id

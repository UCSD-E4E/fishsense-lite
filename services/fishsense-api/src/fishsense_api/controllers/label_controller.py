# pylint: disable=C0121
"""Label Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.server import app

logger = logging.getLogger(__name__)


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

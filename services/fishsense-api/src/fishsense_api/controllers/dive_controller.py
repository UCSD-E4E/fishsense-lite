# pylint: disable=C0121
"""Dive Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.image import Image
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.server import app

logger = logging.getLogger(__name__)


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

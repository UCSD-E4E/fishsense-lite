"""Dive slate controller."""

import logging
from typing import List

from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive_slate import DiveSlate
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/dive-slates/")
async def get_dive_slates(
    session: AsyncSession = Depends(get_async_session),
) -> List[DiveSlate]:
    """Retrieve all dive slates."""
    logger.debug("Retrieving all dive slates")
    query = select(DiveSlate)

    return (await session.exec(query)).all()


@app.put("/api/v1/dive-slates/{dive_slate_id}", status_code=201)
async def put_dive_slate(
    dive_slate_id: int,
    dive_slate: DiveSlate,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a dive slate for a given dive slate ID."""
    logger.debug("Creating or updating dive slate with id=%d", dive_slate_id)
    dive_slate = DiveSlate.model_validate(jsonable_encoder(dive_slate))
    dive_slate.id = dive_slate_id

    dive_slate = await session.merge(dive_slate)
    await session.flush()

    dive_slate_id = dive_slate.id

    return dive_slate_id

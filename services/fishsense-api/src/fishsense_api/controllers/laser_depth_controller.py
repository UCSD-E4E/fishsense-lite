"""Per-image laser-depth endpoints.

Store and read the distance to an image's laser dot. The value is computed on
the data-worker (`fishsense-core`'s `WorldPointHandler` lives there, and only
there — this service has neither it nor numpy), so the API's whole job is to
persist what it is handed and serve it back.
"""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.image import Image
from fishsense_api.models.laser_depth import LaserDepth
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.put("/api/v1/images/{image_id}/laser-depth/", status_code=201)
async def put_laser_depth(
    image_id: int,
    depth: LaserDepth,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update an image's laser depth.

    Upserts on `image_id` (the natural key) so a recompute after a relabel or
    a recalibration overwrites rather than duplicating — `merge` with
    `id=None` keys on the primary key alone and would always INSERT, which is
    how stage 14 once duplicated measurements and how the label writeback
    500'd on retry.
    """
    logger.debug("Upserting laser depth for image with id=%d", image_id)
    depth = LaserDepth.model_validate(jsonable_encoder(depth))
    depth.image_id = image_id

    if depth.id is None:
        existing = (
            await session.exec(
                select(LaserDepth).where(LaserDepth.image_id == image_id)
            )
        ).first()
        if existing is not None:
            depth.id = existing.id

    depth = await session.merge(depth)
    await session.flush()

    return depth.id


@app.get("/api/v1/images/{image_id}/laser-depth/")
async def get_laser_depth(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> LaserDepth:
    """The distance to this image's laser dot.

    404 when there is none — for a single-image lookup the caller asked about
    one specific frame, and an empty body would be indistinguishable from a
    depth of zero.
    """
    logger.debug("Retrieving laser depth for image with id=%d", image_id)
    depth = (
        await session.exec(select(LaserDepth).where(LaserDepth.image_id == image_id))
    ).first()
    if depth is None:
        logger.warning("Laser depth for image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Laser depth not found")
    return depth


@app.get("/api/v1/dives/{dive_id}/laser-depths/")
async def get_laser_depths_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[LaserDepth]:
    """Every laser depth recorded for a dive's images.

    Empty list rather than 404, unlike the single-image read: the compute
    activity fetches this once per dive to see which images it has already
    done, and "none yet" is the normal state on the first run.
    """
    logger.debug("Retrieving laser depths for dive with id=%d", dive_id)
    query = (
        select(LaserDepth)
        .join_from(LaserDepth, Image, LaserDepth.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )
    return (await session.exec(query)).all()

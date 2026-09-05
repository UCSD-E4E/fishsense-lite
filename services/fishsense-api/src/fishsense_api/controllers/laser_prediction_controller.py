# pylint: disable=C0121
"""Laser-prediction endpoints.

Model-predicted laser dots from the fishsense-core LaserDetector stage. The
GPU predict stage upserts one per image; the laser populate step reads a
dive's predictions to seed Label Studio pre-annotations (assisted review).
"""

import logging
from typing import List

from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.image import Image
from fishsense_api.models.laser_prediction import LaserPrediction
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.put("/api/v1/images/{image_id}/laser-prediction/", status_code=201)
async def put_laser_prediction(
    image_id: int,
    prediction: LaserPrediction,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update the model's laser prediction for an image.

    Upserts on `image_id` (the natural key) so a re-run of the predict stage
    overwrites rather than duplicating — merge on `id=None` would always
    INSERT and violate `uq_laser_prediction_image`.
    """
    logger.debug("Upserting laser prediction for image with id=%d", image_id)
    prediction = LaserPrediction.model_validate(jsonable_encoder(prediction))
    prediction.image_id = image_id

    if prediction.id is None:
        existing = (
            await session.exec(
                select(LaserPrediction).where(LaserPrediction.image_id == image_id)
            )
        ).first()
        if existing is not None:
            prediction.id = existing.id

    prediction = await session.merge(prediction)
    await session.flush()

    return prediction.id


@app.get("/api/v1/dives/{dive_id}/laser-predictions/")
async def get_laser_predictions_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[LaserPrediction]:
    """Every model laser prediction for a dive's images.

    Empty list (not 404) when the dive has none — the laser populate step
    treats "no prediction" as "no pre-annotation," not an error.
    """
    logger.debug("Retrieving laser predictions for dive with id=%d", dive_id)
    query = (
        select(LaserPrediction)
        .join_from(LaserPrediction, Image, LaserPrediction.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )
    return (await session.exec(query)).all()

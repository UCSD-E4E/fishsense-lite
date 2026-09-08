# pylint: disable=C0121
"""Slate-prediction endpoints.

Model-predicted dive-slate boards from the fishsense-core slate-detector stage.
The predict stage upserts one per image; the dive-slate populate step reads a
dive's predictions to seed Label Studio pre-annotations (assisted review).
"""

import logging
from typing import List

from fastapi import Depends
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.controllers._prediction_upsert import upsert_prediction
from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.image import Image
from fishsense_api.models.slate_prediction import SlatePrediction
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.put("/api/v1/images/{image_id}/slate-prediction/", status_code=201)
async def put_slate_prediction(
    image_id: int,
    prediction: SlatePrediction,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update the model's slate prediction for an image.

    Upserts on `image_id` (the natural key) so a re-run of the predict stage
    overwrites rather than duplicating — merge on `id=None` would always
    INSERT and violate `uq_slate_prediction_image`.
    """
    logger.debug("Upserting slate prediction for image with id=%d", image_id)
    return await upsert_prediction(session, SlatePrediction, image_id, prediction)


@app.get("/api/v1/dives/{dive_id}/slate-predictions/")
async def get_slate_predictions_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[SlatePrediction]:
    """Every model slate prediction for a dive's images.

    Empty list (not 404) when the dive has none — the dive-slate populate step
    treats "no prediction" as "no pre-annotation," not an error.
    """
    logger.debug("Retrieving slate predictions for dive with id=%d", dive_id)
    query = (
        select(SlatePrediction)
        .join_from(SlatePrediction, Image, SlatePrediction.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )
    return (await session.exec(query)).all()

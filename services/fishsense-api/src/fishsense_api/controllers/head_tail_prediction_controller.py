# pylint: disable=C0121
"""Head/tail-prediction endpoints.

Model-predicted snout/fork keypoints from the GPU head/tail predict stage. The
stage upserts one per image; the head/tail populate step reads a dive's
predictions to seed Label Studio pre-annotations (assisted review).
"""

import logging
from typing import List, Optional

from fastapi import Depends
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.controllers._prediction_upsert import upsert_prediction
from fishsense_api.database import get_async_session
from fishsense_api.models.dive import Dive
from fishsense_api.models.head_tail_prediction import HeadTailPrediction
from fishsense_api.models.image import Image
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.put("/api/v1/images/{image_id}/headtail-prediction/", status_code=201)
async def put_head_tail_prediction(
    image_id: int,
    prediction: HeadTailPrediction,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update the model's head/tail prediction for an image.

    Upserts on `image_id` (the natural key) so a re-run of the predict stage
    overwrites rather than duplicating — merge on `id=None` would always
    INSERT and violate `uq_headtail_prediction_image`.
    """
    logger.debug("Upserting head/tail prediction for image with id=%d", image_id)
    return await upsert_prediction(session, HeadTailPrediction, image_id, prediction)


@app.get("/api/v1/images/{image_id}/headtail-prediction/")
async def get_head_tail_prediction(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> Optional[HeadTailPrediction]:
    """The model's head/tail prediction for one image, or None.

    None (not 404) when absent: "not predicted yet" is an ordinary state the
    cohort selects on, not an error.
    """
    logger.debug("Retrieving head/tail prediction for image with id=%d", image_id)
    return (
        await session.exec(
            select(HeadTailPrediction).where(HeadTailPrediction.image_id == image_id)
        )
    ).first()


@app.get("/api/v1/dives/{dive_id}/headtail-predictions/")
async def get_head_tail_predictions_for_dive(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[HeadTailPrediction]:
    """Every model head/tail prediction for a dive's images.

    Empty list (not 404) when the dive has none — the populate step treats "no
    prediction" as "no pre-annotation," not an error.
    """
    logger.debug("Retrieving head/tail predictions for dive with id=%d", dive_id)
    query = (
        select(HeadTailPrediction)
        .join_from(HeadTailPrediction, Image, HeadTailPrediction.image_id == Image.id)
        .join_from(Image, Dive, Image.dive_id == Dive.id)
        .where(Dive.id == dive_id)
    )
    return (await session.exec(query)).all()

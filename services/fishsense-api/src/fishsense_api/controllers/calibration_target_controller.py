"""Calibration target controller.

The read/upsert surface for `CalibrationTarget` — the planar targets laser
extrinsics can be fitted against. Shaped exactly like `dive_slate_controller`,
because they answer the same question for the two kinds of target; the dive
*link* lives in `dive_controller` beside `set_dive_slate` for the same reason.
"""

import logging
from typing import List

from fastapi import Depends
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.calibration_target import CalibrationTarget
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/calibration-targets/")
async def get_calibration_targets(
    session: AsyncSession = Depends(get_async_session),
) -> List[CalibrationTarget]:
    """Retrieve all calibration targets."""
    logger.debug("Retrieving all calibration targets")
    query = select(CalibrationTarget)

    return (await session.exec(query)).all()


@app.put("/api/v1/calibration-targets/{calibration_target_id}", status_code=201)
async def put_calibration_target(
    calibration_target_id: int,
    calibration_target: CalibrationTarget,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a calibration target.

    Upserts on the id so a re-measured board corrects its own row. Two rows
    for one physical board is the failure worth avoiding: dives would split
    across them, and half the corpus would silently carry the superseded
    scale with nothing recording which was which.
    """
    logger.debug("Creating or updating calibration target id=%d", calibration_target_id)
    calibration_target = CalibrationTarget.model_validate(
        jsonable_encoder(calibration_target)
    )
    calibration_target.id = calibration_target_id

    calibration_target = await session.merge(calibration_target)
    await session.flush()

    return calibration_target.id

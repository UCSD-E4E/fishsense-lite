"""Camera Controller for FishSense API."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.camera import Camera
from fishsense_api.models.camera_intrinsics import CameraIntrinsics
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/cameras/")
async def get_cameras(
    session: AsyncSession = Depends(get_async_session),
) -> List[Camera]:
    """Retrieve all cameras."""
    logger.debug("Retrieving all cameras")
    query = select(Camera)

    return (await session.exec(query)).all()


@app.get("/api/v1/cameras/{camera_id}")
async def get_camera(
    camera_id: int, session: AsyncSession = Depends(get_async_session)
) -> Camera | None:
    """Retrieve a camera by its ID."""
    logger.debug("Retrieving camera with id=%d", camera_id)
    query = select(Camera).where(Camera.id == camera_id)

    camera = (await session.exec(query)).first()
    if camera is None:
        logger.warning("Camera with id=%d not found", camera_id)
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@app.get("/api/v1/cameras/{camera_id}/intrinsics/")
async def get_camera_intrinsics(
    camera_id: int, session: AsyncSession = Depends(get_async_session)
) -> CameraIntrinsics | None:
    """Retrieve camera intrinsics for a given camera ID."""
    logger.debug("Retrieving intrinsics for camera with id=%d", camera_id)
    query = select(CameraIntrinsics).where(CameraIntrinsics.camera_id == camera_id)

    camera_intrinsics = (await session.exec(query)).first()
    if camera_intrinsics is None:
        logger.warning("Camera intrinsics for camera with id=%d not found", camera_id)
        raise HTTPException(status_code=404, detail="Camera intrinsics not found")
    return camera_intrinsics


@app.put("/api/v1/cameras/{camera_id}/intrinsics/", status_code=201)
async def put_camera_intrinsics(
    camera_id: int,
    intrinsics: CameraIntrinsics,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update camera intrinsics for a given camera ID."""
    logger.debug("Creating or updating intrinsics for camera with id=%d", camera_id)
    intrinsics.camera_id = camera_id

    intrinsics = await session.merge(intrinsics)
    await session.flush()

    intrinsics_id = intrinsics.id

    return intrinsics_id

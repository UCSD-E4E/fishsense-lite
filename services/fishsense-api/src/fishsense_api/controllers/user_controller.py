"""Controller for user-related API endpoints."""

import logging
from typing import List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.user import User
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/users/")
async def get_users(session: AsyncSession = Depends(get_async_session)) -> List[User]:
    """Retrieve all users."""
    logger.debug("Retrieving all users")

    query = select(User)
    results = await session.exec(query)
    return results.all()


@app.get("/api/v1/users/{user_id}")
async def get_user(
    user_id: int, session: AsyncSession = Depends(get_async_session)
) -> User | None:
    """Retrieve a user by their ID."""
    logger.debug("Retrieving user with id=%d", user_id)

    query = select(User).where(User.id == user_id)
    result = await session.exec(query)
    user = result.first()
    logger.debug("Query result for user with id=%d: %s", user_id, user)
    if user is None:
        logger.warning("User with id=%d not found; raising HTTPException 404", user_id)
        raise HTTPException(status_code=404, detail="User not found")
    logger.debug("User with id=%d found successfully", user_id)
    return user


@app.get("/api/v1/users/label-studio/{label_studio_id}")
async def get_user_by_label_studio_id(
    label_studio_id: int, session: AsyncSession = Depends(get_async_session)
) -> User | None:
    """Retrieve a user by their Label Studio ID."""
    logger.debug("Retrieving user with label_studio_id=%d", label_studio_id)

    query = select(User).where(User.label_studio_id == label_studio_id)
    result = await session.exec(query)
    user = result.first()
    logger.debug(
        "Query result for user with label_studio_id=%d: %s", label_studio_id, user
    )
    if user is None:
        logger.warning(
            "User with label_studio_id=%d not found; raising HTTPException 404",
            label_studio_id,
        )
        raise HTTPException(status_code=404, detail="User not found")
    logger.debug("User with label_studio_id=%d found successfully", label_studio_id)
    return user


@app.get("/api/v1/users/email/{email}")
async def get_user_by_email(
    email: str, session: AsyncSession = Depends(get_async_session)
) -> User | None:
    """Retrieve a user by their email."""
    logger.debug("Retrieving user by email")

    query = select(User).where(User.email == email)
    result = await session.exec(query)
    user = result.first()
    logger.debug("Query result for user by email: %s", user)
    if user is None:
        logger.warning("User not found for provided email; raising HTTPException 404")
        raise HTTPException(status_code=404, detail="User not found")
    logger.debug("User found successfully by email")
    return user


@app.post("/api/v1/users/", status_code=201)
async def create_user(
    user: User, session: AsyncSession = Depends(get_async_session)
) -> int:
    """Create a new user."""
    logger.debug("Creating a new user with label_studio_id=%s", user.label_studio_id)
    try:
        user = User.model_validate(jsonable_encoder(user))
        logger.debug(
            "Validated user model with label_studio_id=%s", user.label_studio_id
        )
        session.add(user)
        logger.debug("User added to session; flushing to persist")
        await session.flush()
        await session.refresh(user)
        logger.debug("User created successfully with id=%s", user.id)
        return user.id
    except Exception:
        logger.exception("Failed to create user")
        raise


@app.put("/api/v1/users/{user_id}", status_code=201)
async def create_or_update_user(
    user_id: int,
    user: User,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update a user by their ID."""
    logger.debug("Creating or updating user with id=%d", user_id)
    user = User.model_validate(jsonable_encoder(user))
    user.id = user_id

    user = await session.merge(user)
    await session.flush()

    user_id = user.id

    return user_id

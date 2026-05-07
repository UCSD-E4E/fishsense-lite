# pylint: disable=unused-import
"""Database interaction module for FishSense API Workflow Worker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

import sqlalchemy as sa
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.config import pg_connection_string
from fishsense_api.models.camera import Camera
from fishsense_api.models.camera_intrinsics import CameraIntrinsics
from fishsense_api.models.dive import Dive
from fishsense_api.models.dive_frame_cluster import (
    DiveFrameCluster,
    DiveFrameClusterImageMapping,
)
from fishsense_api.models.dive_slate import DiveSlate
from fishsense_api.models.dive_slate_label import DiveSlateLabel
from fishsense_api.models.fish import Fish
from fishsense_api.models.head_tail_label import HeadTailLabel
from fishsense_api.models.image import Image
from fishsense_api.models.label_studio_sync_cursor import LabelStudioSyncCursor
from fishsense_api.models.laser_extrinsics import LaserExtrinsics
from fishsense_api.models.laser_label import LaserLabel
from fishsense_api.models.measurement import Measurement
from fishsense_api.models.species import Species
from fishsense_api.models.species_label import SpeciesLabel
from fishsense_api.models.user import User

_log = logging.getLogger(__name__)


class Database:
    # pylint: disable=too-few-public-methods

    """Database interaction class for FishSense API Workflow Worker."""

    def __init__(self, database_url: str):
        self.engine = create_async_engine(
            database_url,
            pool_size=5,  # Maintain a minimum of 5 connections in the pool
            max_overflow=2,  # Allow up to 2 additional connections beyond pool_size
            pool_timeout=30,  # Wait up to 30 seconds for a connection before raising an error
            pool_recycle=3600,  # Recycle connections after 1 hour to prevent stale connections
            pool_pre_ping=True,  # pre-ping to check if connections are alive before using them
        )

    async def init_database(self, conn: AsyncSession) -> None:
        """Initialize the database by creating all tables."""
        await conn.run_sync(SQLModel.metadata.create_all)

    async def dispose(self) -> None:
        """Dispose of the database engine."""
        await self.engine.dispose()


async def _has_alembic_version_table() -> bool:
    """Whether the current DB already has the alembic_version table.

    Used as the fresh-vs-existing DB signal by `run_alembic_upgrade`.
    Async because asyncpg is the only Postgres driver in the runtime
    image; spinning up a sync engine would require psycopg2 as a
    new dep.
    """
    engine = create_async_engine(pg_connection_string())
    try:
        async with engine.connect() as conn:
            return await conn.run_sync(
                lambda c: sa.inspect(c).has_table("alembic_version")
            )
    finally:
        await engine.dispose()


def run_alembic_upgrade() -> None:
    """Apply pending migrations OR stamp head on a fresh DB.

    Invoked from the FastAPI lifespan so a deploy that ships a new
    migration (e.g. the `dive_pipeline_status` view) doesn't require
    a manual operator step.

    On a **fresh DB** (no `alembic_version` table), the lifespan's
    prior `SQLModel.metadata.create_all` call has just populated the
    full ORM-defined schema. Running the historical migration tail
    on top would crash on every pre-existing column / table /
    constraint — `add_column`, `create_table`, and
    `create_unique_constraint` ops in the historical migrations
    weren't written idempotent against `create_all`. Stamp head
    instead to mark the DB as fully migrated without doing any DDL.

    On an **existing DB** (alembic_version present), upgrade as
    normal — any new migrations land on top. New migrations going
    forward must still be idempotent against `create_all` because
    `create_all` keeps running before this on every restart.

    The Config is built programmatically because `alembic.ini` does
    NOT ship inside the wheel — `uv sync --no-editable` only installs
    the package source under `site-packages/fishsense_api/`. Locating
    `script_location` relative to this module's path keeps the
    migration scripts findable regardless of the runtime's CWD.

    Sync API; callers should offload to a worker thread
    (`asyncio.to_thread`) to avoid blocking the event loop, since
    `command.upgrade` opens its own engine and runs DDL synchronously
    via the async-engine bridge in `alembic/env.py`. The
    `_has_alembic_version_table` check uses `asyncio.run` because we
    are already off the FastAPI event-loop thread.
    """
    cfg = AlembicConfig()
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parent / "alembic"),
    )

    if asyncio.run(_has_alembic_version_table()):
        _log.info("alembic_version present; running upgrade head")
        alembic_command.upgrade(cfg, "head")
        _log.info("alembic upgrade complete")
    else:
        _log.info(
            "alembic_version missing; stamping head (fresh DB after create_all)"
        )
        alembic_command.stamp(cfg, "head")


_session_factory: sessionmaker | None = None  # pylint: disable=invalid-name


def setup_database() -> Database:
    """Construct the Database and session factory from settings.

    Call once at application startup (e.g., FastAPI lifespan) before any
    request handler invokes ``get_async_session``.
    """
    global _session_factory  # pylint: disable=global-statement
    database = Database(pg_connection_string())
    _session_factory = sessionmaker(
        database.engine, class_=AsyncSession, expire_on_commit=False
    )
    return database


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """A context manager for getting a session.

    Yields:
        AsyncSession: An asynchronous database session.
    """
    if _session_factory is None:
        raise RuntimeError(
            "Database not initialized; call setup_database() first"
        )
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            _log.exception("session rolled back due to unhandled exception")
            await session.rollback()
            raise

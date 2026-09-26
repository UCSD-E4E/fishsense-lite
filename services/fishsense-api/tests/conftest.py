"""Fixtures pytest injects into every test module in this package.

The in-memory sqlite `session` lives here rather than in `tests_support.db`
so no test module has to import it: pytest resolves fixtures by name, and an
imported-but-never-called fixture reads as an unused import to every static
analyser. The row builders stay in `tests_support.db`, since those are
ordinary functions that tests call explicitly.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    """A session over a fresh in-memory schema, disposed afterwards.

    Importing `fishsense_api.database` is load-bearing, not an unused import:
    it is the model registry, and without it `SQLModel.metadata.create_all`
    sees only whichever models the test module happened to import.
    """
    # pylint: disable=unused-import
    #   Imported for its side effect: it is the model registry.
    import fishsense_api.database  # noqa: F401

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as open_session:
        yield open_session
    await engine.dispose()


@pytest.fixture
async def http(session):
    """The app over the in-memory session, without running the real lifespan.

    Entering `TestClient`'s context manager would run `lifespan`, which calls
    `create_all` against Postgres and then `run_alembic_upgrade`.
    """
    from fastapi.testclient import TestClient

    from tests_support.app import seed_placeholder_settings

    seed_placeholder_settings()

    import fishsense_api.controllers  # noqa: F401  pylint: disable=unused-import
    from fishsense_api.database import get_async_session
    from fishsense_api.server import app

    async def _override():
        yield session

    app.dependency_overrides[get_async_session] = _override
    yield TestClient(app)
    app.dependency_overrides.pop(get_async_session, None)

"""Fish identity endpoints: lookup + upsert on the `name` natural key.

Physical fish models are keyed by `name` so the same model resolves to one
Fish across dives. Stage 14 resolves-or-creates that Fish, so:
  * `GET /api/v1/fish/by-name/{name}` must find it (or 404 cleanly), and
  * `POST /api/v1/fish` must UPSERT on `name` — a blind insert would hit the
    `uq_fish_name` constraint and 500 the second time a model is measured
    (this repo has been bitten by blind-merge duplicate-key 500s before).
Real fish keep `name=None` and must never collide on the nullable-unique key.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _fish(name=None, species_id=None, fish_id=None):
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel

    return Fish(id=fish_id, name=name, species_id=species_id)


async def _get_by_name(session, name: str):
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        get_fish_by_name,
    )

    return await get_fish_by_name(name, session=session)


async def _post(session, fish):
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        post_fish,
    )

    return await post_fish(fish, session=session)


# ── GET /api/v1/fish/by-name/{name} ──────────────────────────────────


async def test_get_fish_by_name_returns_match(session):
    session.add_all([_fish(name="Grouper"), _fish(name="Shark")])
    await session.flush()

    fish = await _get_by_name(session, "Grouper")

    assert fish is not None
    assert fish.name == "Grouper"


async def test_get_fish_by_name_404s_when_absent(session):
    with pytest.raises(Exception):  # HTTPException 404
        await _get_by_name(session, "Nonexistent Model")


# ── POST /api/v1/fish upserts on name ────────────────────────────────


async def test_post_fish_creates_named(session):
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel

    new_id = await _post(session, _fish(name="Grouper"))
    await session.flush()

    rows = (await session.exec(select(Fish))).all()
    assert len(rows) == 1
    assert new_id is not None
    assert rows[0].name == "Grouper"


async def test_post_fish_upserts_on_name(session):
    """Second POST of the same model name must not insert a duplicate."""
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel

    first = await _post(session, _fish(name="Grouper"))
    await session.flush()
    second = await _post(session, _fish(name="Grouper"))
    await session.flush()

    rows = (await session.exec(select(Fish))).all()
    assert len(rows) == 1, "re-posting the same name must upsert, not insert"
    assert first == second


async def test_post_fish_null_names_do_not_collide(session):
    """Real fish (name=None) must both insert under the nullable-unique key."""
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel

    await _post(session, _fish(name=None, species_id=None))
    await session.flush()
    await _post(session, _fish(name=None, species_id=None))
    await session.flush()

    rows = (await session.exec(select(Fish))).all()
    assert len(rows) == 2

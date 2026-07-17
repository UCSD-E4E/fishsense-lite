"""Measurement read + write semantics.

Two things are pinned here, both in service of making stage 14
(`measure_fish_activity`) safe to re-run:

* `GET /api/v1/dives/{dive_id}/measurements` — the dive-scoped read the
  activity uses to skip images it has already measured. Without it the
  activity has no way to ask "did I already do this one?".
* `POST /api/v1/fish/{fish_id}/measurements` upserts on
  `(image_id, fish_id)` rather than always inserting. The old behavior
  (`session.merge` with `id=None`) keyed on the primary key only, so a
  re-run on a partially-measured dive silently duplicated every
  already-measured image.

Mirrors the in-memory-sqlite fixture style of
test_dives_with_complete_laser_labeling_endpoint.py — testing query
composition and write semantics, not referential integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


def _dive(dive_id: int):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
    )


def _image(image_id: int, dive_id: int):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"img-{image_id:032d}"[:32],
        dive_id=dive_id,
    )


def _fish(fish_id: int):
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel

    return Fish(id=fish_id, species_id=None)


def _measurement(measurement_id: int | None, image_id: int, fish_id: int, length_m: float):
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    return Measurement(
        id=measurement_id,
        image_id=image_id,
        fish_id=fish_id,
        length_m=length_m,
    )


async def _seed_dive(session, dive_id: int, image_ids: list[int]):
    session.add_all([_dive(dive_id)])
    await session.flush()
    session.add_all([_image(i, dive_id) for i in image_ids])
    await session.flush()


async def _get_measurements(session, dive_id: int):
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        get_measurements_for_dive,
    )

    return await get_measurements_for_dive(dive_id, session=session)


async def _post(session, fish_id: int, measurement):
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        post_measurement,
    )

    return await post_measurement(fish_id, measurement, session=session)


# ── GET /api/v1/dives/{dive_id}/measurements ─────────────────────────


async def test_get_measurements_returns_only_that_dives_rows(session):
    await _seed_dive(session, 1, [11, 12])
    await _seed_dive(session, 2, [21])
    session.add_all([_fish(101), _fish(102), _fish(201)])
    await session.flush()
    session.add_all(
        [
            _measurement(1, 11, 101, 0.30),
            _measurement(2, 12, 102, 0.40),
            _measurement(3, 21, 201, 0.50),  # different dive — must not leak
        ]
    )
    await session.flush()

    rows = await _get_measurements(session, 1)

    assert sorted(m.image_id for m in rows) == [11, 12]


async def test_get_measurements_404s_when_dive_has_none(session):
    """Empty reads as 404, matching get_species_labels_for_dive."""
    await _seed_dive(session, 1, [11])

    with pytest.raises(Exception):  # HTTPException 404 — no measurements yet
        await _get_measurements(session, 1)


# ── POST is idempotent on (image_id, fish_id) ────────────────────────


async def test_post_measurement_creates_row(session):
    await _seed_dive(session, 1, [11])
    session.add_all([_fish(101)])
    await session.flush()

    new_id = await _post(session, 101, _measurement(None, 11, 101, 0.30))
    await session.flush()

    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    rows = (await session.exec(select(Measurement))).all()
    assert len(rows) == 1
    assert new_id is not None
    assert rows[0].length_m == pytest.approx(0.30)


async def test_post_measurement_twice_does_not_duplicate(session):
    """The stage-14 re-run case: same (image, fish) must not stack rows."""
    await _seed_dive(session, 1, [11])
    session.add_all([_fish(101)])
    await session.flush()

    first_id = await _post(session, 101, _measurement(None, 11, 101, 0.30))
    await session.flush()
    second_id = await _post(session, 101, _measurement(None, 11, 101, 0.31))
    await session.flush()

    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    rows = (await session.exec(select(Measurement))).all()
    assert len(rows) == 1, "re-posting the same (image, fish) must upsert, not insert"
    assert first_id == second_id
    assert rows[0].length_m == pytest.approx(0.31), "latest length wins"


async def test_post_measurement_distinct_fish_on_same_image_both_persist(session):
    """Two fish in one frame are legitimately two measurements."""
    await _seed_dive(session, 1, [11])
    session.add_all([_fish(101), _fish(102)])
    await session.flush()

    await _post(session, 101, _measurement(None, 11, 101, 0.30))
    await session.flush()
    await _post(session, 102, _measurement(None, 11, 102, 0.40))
    await session.flush()

    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    rows = (await session.exec(select(Measurement))).all()
    assert len(rows) == 2


async def test_post_measurement_binds_fish_id_from_path(session):
    """fish_id comes from the URL, not the body."""
    await _seed_dive(session, 1, [11])
    session.add_all([_fish(101)])
    await session.flush()

    await _post(session, 101, _measurement(None, 11, 999, 0.30))
    await session.flush()

    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    rows = (await session.exec(select(Measurement))).all()
    assert rows[0].fish_id == 101

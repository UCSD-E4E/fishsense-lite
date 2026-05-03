"""SQL behavior of `GET /api/v1/labels/laser/dives-with-complete-labeling`.

Pins the predicate: a dive qualifies iff every non-superseded LaserLabel
for one of its images has `completed=True` AND at least one such label
exists. Mirrors the in-memory-sqlite fixture style of
test_select_next_dive_endpoints.py — testing query composition, not
referential integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
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


def _laser_label(
    label_id: int,
    image_id: int,
    *,
    completed: bool | None,
    superseded: bool = False,
    project_id: int | None = 42,
):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    return LaserLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
    )


async def _call(session):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_dives_with_complete_laser_labeling,
    )

    return await get_dives_with_complete_laser_labeling(session=session)


async def test_dive_with_only_completed_labels_qualifies(session):
    session.add_all([_dive(1)])
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True),
            _laser_label(102, 12, completed=True),
        ]
    )
    await session.flush()

    assert sorted(await _call(session)) == [1]


async def test_dive_with_any_incomplete_label_excluded(session):
    session.add_all([_dive(1)])
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True),
            _laser_label(102, 12, completed=False),
        ]
    )
    await session.flush()

    assert await _call(session) == []


async def test_dive_with_null_completed_label_excluded(session):
    """`completed IS NULL` is treated as 'incomplete' to match the
    discovery-side predicate `(completed = false OR IS NULL)`."""
    session.add_all([_dive(1)])
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True),
            _laser_label(102, 12, completed=None),
        ]
    )
    await session.flush()

    assert await _call(session) == []


async def test_superseded_incomplete_label_does_not_block(session):
    """Once a label is superseded it shouldn't count against the
    completeness check — that's what the column is for."""
    session.add_all([_dive(1)])
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True, project_id=42),
            _laser_label(102, 12, completed=True, project_id=42),
            # Older incomplete row in a different project, now superseded.
            # Different project_id so the (image_id, project_id) UNIQUE
            # constraint doesn't trip on the in-memory sqlite fixture.
            _laser_label(103, 11, completed=False, superseded=True, project_id=99),
        ]
    )
    await session.flush()

    assert sorted(await _call(session)) == [1]


async def test_dive_with_zero_labels_excluded(session):
    """A dive with no laser labels at all has nothing to validate."""
    session.add_all([_dive(1)])
    await session.flush()
    session.add_all([_image(11, 1)])
    await session.flush()

    assert await _call(session) == []


async def test_only_superseded_labels_excluded(session):
    """All labels superseded → no non-superseded population to
    validate → dive is excluded."""
    session.add_all([_dive(1)])
    await session.flush()
    session.add_all([_image(11, 1)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True, superseded=True),
        ]
    )
    await session.flush()

    assert await _call(session) == []


async def test_mixed_dives_only_complete_returned(session):
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all(
        [_image(11, 1), _image(21, 2), _image(22, 2), _image(31, 3)]
    )
    await session.flush()
    session.add_all(
        [
            # Dive 1: complete.
            _laser_label(101, 11, completed=True),
            # Dive 2: one complete, one incomplete -> excluded.
            _laser_label(201, 21, completed=True),
            _laser_label(202, 22, completed=False),
            # Dive 3: complete.
            _laser_label(301, 31, completed=True),
        ]
    )
    await session.flush()

    assert sorted(await _call(session)) == [1, 3]

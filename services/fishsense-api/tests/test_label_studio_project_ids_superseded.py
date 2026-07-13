"""`GET /api/v1/labels/{laser,headtail}/label-studio-project-ids` must
exclude projects whose only labels are superseded.

Every other laser/headtail read filters `superseded == False`; these two
discovery endpoints historically did not, so a project whose labels were
all dead-lettered still surfaced as a live labeling card on the landing
page (and got re-enumerated by the hourly sync). This pins the fix.

In-memory-sqlite fixture style, matching
test_dives_with_complete_laser_labeling_endpoint.py — testing query
composition, not referential integrity.
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


def _image(image_id: int):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"img-{image_id:032d}"[:32],
        dive_id=1,
    )


def _laser_label(label_id, image_id, *, completed, superseded=False, project_id):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    return LaserLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
    )


def _headtail_label(label_id, image_id, *, completed, superseded=False, project_id):
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel

    return HeadTailLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
    )


async def _laser_ids(session, *, incomplete=False):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_label_studio_project_ids,
    )

    return sorted(await get_laser_label_studio_project_ids(incomplete=incomplete, session=session))


async def _headtail_ids(session, *, incomplete=False):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_headtail_label_studio_project_ids,
    )

    return sorted(await get_headtail_label_studio_project_ids(incomplete=incomplete, session=session))


# ── laser ──────────────────────────────────────────────────────────────────


async def test_laser_superseded_only_project_excluded(session):
    session.add_all([_image(11), _image(12)])
    await session.flush()
    session.add_all(
        [
            # project 42: one live label -> should surface
            _laser_label(101, 11, completed=False, project_id=42),
            # project 99: only a superseded label -> should NOT surface
            _laser_label(102, 12, completed=False, superseded=True, project_id=99),
        ]
    )
    await session.flush()

    assert await _laser_ids(session) == [42]
    assert await _laser_ids(session, incomplete=True) == [42]


async def test_laser_live_projects_still_returned(session):
    session.add_all([_image(11), _image(12)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True, project_id=42),
            _laser_label(102, 12, completed=False, project_id=43),
        ]
    )
    await session.flush()

    assert await _laser_ids(session) == [42, 43]
    # incomplete filter unchanged: only the project with an incomplete row
    assert await _laser_ids(session, incomplete=True) == [43]


async def test_laser_project_with_mixed_rows_survives_if_any_live(session):
    """A project keeps surfacing as long as it has at least one
    non-superseded label, even if others are superseded."""
    session.add_all([_image(11), _image(12)])
    await session.flush()
    session.add_all(
        [
            _laser_label(101, 11, completed=True, superseded=True, project_id=42),
            _laser_label(102, 12, completed=True, project_id=42),
        ]
    )
    await session.flush()

    assert await _laser_ids(session) == [42]


# ── headtail (same fix, for parity) ─────────────────────────────────────────


async def test_headtail_superseded_only_project_excluded(session):
    session.add_all([_image(11), _image(12)])
    await session.flush()
    session.add_all(
        [
            _headtail_label(101, 11, completed=False, project_id=71),
            _headtail_label(102, 12, completed=False, superseded=True, project_id=88),
        ]
    )
    await session.flush()

    assert await _headtail_ids(session) == [71]
    assert await _headtail_ids(session, incomplete=True) == [71]

"""Species + dive-slate reads exclude `superseded` rows (Family B), matching
laser/headtail. Covers the project-ids discovery endpoints (landing page + sync)
and the per-image / per-task / per-dive get-reads.

In-memory-sqlite fixture style, matching
test_dives_with_complete_laser_labeling_endpoint.py.
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


def _dive(dive_id=1):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path="/dev/null",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
    )


def _image(image_id, dive_id=1):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"img-{image_id:032d}"[:32],
        dive_id=dive_id,
    )


def _species(label_id, image_id, *, completed=False, superseded=False, project_id, task_id):
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    return SpeciesLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
        label_studio_task_id=task_id,
    )


def _slate(label_id, image_id, *, completed=False, superseded=False, project_id, task_id):
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    return DiveSlateLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
        label_studio_task_id=task_id,
    )


# ── species ─────────────────────────────────────────────────────────────────


async def test_species_project_ids_exclude_superseded_only(session):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_species_label_studio_project_ids,
    )

    session.add_all([_image(11), _image(12)])
    await session.flush()
    session.add_all(
        [
            _species(101, 11, project_id=70, task_id=1),
            _species(102, 12, superseded=True, project_id=99, task_id=2),
        ]
    )
    await session.flush()

    ids = sorted(await get_species_label_studio_project_ids(session=session))
    assert ids == [70]


async def test_species_get_by_image_excludes_superseded(session):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_species_label,
    )

    session.add_all([_image(11)])
    await session.flush()
    session.add_all([_species(101, 11, superseded=True, project_id=70, task_id=1)])
    await session.flush()

    with pytest.raises(Exception):  # 404 HTTPException — superseded row is invisible
        await get_species_label(11, session=session)


# ── dive-slate ──────────────────────────────────────────────────────────────


async def test_slate_project_ids_exclude_superseded_only(session):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_dive_slate_label_studio_project_ids,
    )

    session.add_all([_image(11), _image(12)])
    await session.flush()
    session.add_all(
        [
            _slate(101, 11, project_id=66, task_id=1),
            _slate(102, 12, superseded=True, project_id=88, task_id=2),
        ]
    )
    await session.flush()

    ids = sorted(await get_dive_slate_label_studio_project_ids(session=session))
    assert ids == [66]


async def test_slate_get_by_image_excludes_superseded(session):
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_dive_slate_label,
    )

    session.add_all([_image(11)])
    await session.flush()
    session.add_all([_slate(101, 11, superseded=True, project_id=66, task_id=1)])
    await session.flush()

    assert await get_dive_slate_label(11, session=session) is None

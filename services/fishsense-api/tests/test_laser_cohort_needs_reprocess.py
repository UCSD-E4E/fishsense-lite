"""Stage 0.1's cohort selects flagged dives, not only unlabelled ones.

The cohort's whole predicate used to be "has a canonical image with no
non-sentinel LaserLabel row", which by construction goes false the moment
populate seeds a row — that is what makes it drain, and it is also why an
overlay change reaches no image that has already been preprocessed.

`needs_reprocess` is the second way in. Both halves have to move together:
a selector that ignores the flag never picks the dive up, and a selector
that honours it while the *resolver* ignores it picks the dive, stages its
raw `.ORF`s from the NAS, resolves zero images and does it again next hour.
That mismatch is the standing failure mode in this file's neighbourhood —
CLAUDE.md requires resolvers to mirror the selector predicate exactly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _dive(session, dive_id, *, labelled=True, flagged=False, canonical=True):
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 8, 1, tzinfo=timezone.utc)
    session.add(
        Dive(id=dive_id, path=f"/d{dive_id}", priority=Priority.HIGH, dive_datetime=when)
    )
    image_id = dive_id * 100
    session.add(
        Image(
            id=image_id,
            dive_id=dive_id,
            path=f"/d{dive_id}/a.ORF",
            taken_datetime=when,
            checksum=f"c{dive_id}",
            is_canonical=canonical,
        )
    )
    if labelled:
        session.add(
            LaserLabel(
                image_id=image_id,
                label_studio_project_id=1,
                completed=True,
                needs_reprocess=flagged,
            )
        )
    await session.commit()
    return image_id


async def _select(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_laser_preprocessing,
    )

    return await select_next_for_laser_preprocessing(session=session)


async def test_fully_labelled_unflagged_dive_is_not_selected(session):
    """The pre-existing behaviour, unchanged -- this is what makes the
    cohort drain."""
    await _dive(session, 1, labelled=True, flagged=False)
    assert await _select(session) is None


async def test_flagged_dive_is_selected_though_fully_labelled(session):
    await _dive(session, 1, labelled=True, flagged=True)
    assert await _select(session) == 1


async def test_unlabelled_dive_is_still_selected(session):
    await _dive(session, 1, labelled=False)
    assert await _select(session) == 1


async def test_lowest_id_wins_across_both_reasons(session):
    await _dive(session, 1, labelled=True, flagged=False)
    await _dive(session, 2, labelled=True, flagged=True)
    await _dive(session, 3, labelled=False)
    assert await _select(session) == 2


async def test_flag_on_a_non_canonical_image_does_not_select(session):
    """Only canonical frames are ever preprocessed, so a flag on a duplicate
    row would select a dive the resolver then finds no work for -- the exact
    selector/resolver mismatch this pairing exists to avoid."""
    await _dive(session, 1, labelled=True, flagged=True, canonical=False)
    assert await _select(session) is None


async def test_clearing_the_flag_drains_the_dive(session):
    from fishsense_api.controllers.label_controller import (
        clear_laser_labels_needs_reprocess,
    )

    await _dive(session, 1, labelled=True, flagged=True)
    assert await _select(session) == 1

    await clear_laser_labels_needs_reprocess(dive_id=1, session=session)
    await session.commit()
    assert await _select(session) is None

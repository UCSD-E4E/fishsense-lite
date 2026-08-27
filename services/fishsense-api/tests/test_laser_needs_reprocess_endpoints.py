"""Setting and clearing `needs_reprocess` for a dive's laser labels.

The pair exists because the flag has to be both *raised* by an operator and
*lowered* by the pipeline. Lowering is the half that matters: stage 0.1's
cohort now selects flagged images, so a flag nothing clears would hold its
dive in the cohort forever, re-staging the dive's raw `.ORF`s from the NAS
every hour and blocking every higher-id dive behind it. That is not
hypothetical -- it is exactly how prod dive 60 wedged dives 84/465/471 until
2026-08-04, and how dives 59/439 were stranded by ALLOW_DUPLICATE_FAILED_ONLY.

Scoped to canonical images: the same physical frame lives under several dive
rows and only the canonical copy is ever preprocessed, so flagging the others
would raise a flag no cohort can ever lower.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
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


async def _seed(session):
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 8, 1, tzinfo=timezone.utc)
    session.add(Dive(id=1, path="/d1", priority=Priority.HIGH, dive_datetime=when))
    session.add(Dive(id=2, path="/d2", priority=Priority.HIGH, dive_datetime=when))
    # dive 1: two canonical images (labelled) + one non-canonical duplicate
    session.add(Image(id=10, dive_id=1, path="/d1/a.ORF", taken_datetime=when, checksum="a", is_canonical=True))
    session.add(Image(id=11, dive_id=1, path="/d1/b.ORF", taken_datetime=when, checksum="b", is_canonical=True))
    session.add(Image(id=12, dive_id=1, path="/d1/c.ORF", taken_datetime=when, checksum="c", is_canonical=False))
    # dive 2: must be untouched by a dive-1 call
    session.add(Image(id=20, dive_id=2, path="/d2/d.ORF", taken_datetime=when, checksum="d", is_canonical=True))
    for image_id in (10, 11, 12, 20):
        session.add(
            LaserLabel(image_id=image_id, label_studio_project_id=1, completed=True)
        )
    await session.commit()


async def _flags(session, image_ids):
    from fishsense_api.models.laser_label import LaserLabel

    rows = (
        await session.exec(select(LaserLabel).where(LaserLabel.image_id.in_(image_ids)))
    ).all()
    return {r.image_id: r.needs_reprocess for r in rows}


async def test_put_flags_the_dives_canonical_labels(session):
    from fishsense_api.controllers.label_controller import (
        set_laser_labels_needs_reprocess,
    )

    await _seed(session)
    count = await set_laser_labels_needs_reprocess(dive_id=1, session=session)
    await session.commit()

    assert count == 2
    assert await _flags(session, [10, 11]) == {10: True, 11: True}


async def test_put_skips_non_canonical_images(session):
    from fishsense_api.controllers.label_controller import (
        set_laser_labels_needs_reprocess,
    )

    await _seed(session)
    await set_laser_labels_needs_reprocess(dive_id=1, session=session)
    await session.commit()

    assert await _flags(session, [12]) == {12: False}


async def test_put_does_not_touch_other_dives(session):
    from fishsense_api.controllers.label_controller import (
        set_laser_labels_needs_reprocess,
    )

    await _seed(session)
    await set_laser_labels_needs_reprocess(dive_id=1, session=session)
    await session.commit()

    assert await _flags(session, [20]) == {20: False}


async def test_delete_clears_the_flag(session):
    from fishsense_api.controllers.label_controller import (
        clear_laser_labels_needs_reprocess,
        set_laser_labels_needs_reprocess,
    )

    await _seed(session)
    await set_laser_labels_needs_reprocess(dive_id=1, session=session)
    await session.commit()

    count = await clear_laser_labels_needs_reprocess(dive_id=1, session=session)
    await session.commit()

    assert count == 2
    assert await _flags(session, [10, 11]) == {10: False, 11: False}


async def test_both_are_idempotent(session):
    """The parent workflow clears on every firing, including firings where
    nothing was flagged -- that must be a cheap no-op, not an error."""
    from fishsense_api.controllers.label_controller import (
        clear_laser_labels_needs_reprocess,
        set_laser_labels_needs_reprocess,
    )

    await _seed(session)
    assert await set_laser_labels_needs_reprocess(dive_id=1, session=session) == 2
    assert await set_laser_labels_needs_reprocess(dive_id=1, session=session) == 2
    await session.commit()
    assert await clear_laser_labels_needs_reprocess(dive_id=1, session=session) == 2
    assert await clear_laser_labels_needs_reprocess(dive_id=1, session=session) == 2
    await session.commit()
    assert await _flags(session, [10, 11]) == {10: False, 11: False}


async def test_unknown_dive_is_a_no_op_not_a_404(session):
    """Called unconditionally by the parent workflow on every firing; a 404
    would fail the workflow for a dive that simply has no laser labels."""
    from fishsense_api.controllers.label_controller import (
        clear_laser_labels_needs_reprocess,
    )

    await _seed(session)
    assert await clear_laser_labels_needs_reprocess(dive_id=999, session=session) == 0

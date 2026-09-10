"""`SpeciesLabel.fish_angle_degrees` — the reviewed angle, in degrees.

`fish_angle_category` cannot hold an angle test's independent variable. It is
the Label Studio taxonomy leaf, and its top bucket is the open interval
`x > 15°` — so on prod dive 87 (`083123_Fish Deg Angle Tests_FSL06`) groups
G10 through G14 are five *different* commanded angles that all read
`x > 15°`. Anything downstream asking "how does length error vary with angle?"
gets five distinct populations collapsed into one label.

So this is a second, numeric column rather than a reinterpretation of the
first. The two have different owners, which is the reason not to merge them:
`fish_angle_category` is a labeler's judgment and the hourly species sync is
its single writer, while `fish_angle_degrees` is the reviewed angle from the
test protocol and the sync must never touch it.

NULL is meaningful here and is deliberately *not* backfilled to 0.0 —
unreviewed and "reviewed as zero degrees" are different states, and 0° is a
real value in this sweep. That is the opposite call from `needs_reprocess`,
which took `server_default=false` precisely because its NULL would have made a
cohort predicate skip every pre-existing row.
"""

from __future__ import annotations

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


def _model():
    from fishsense_api.models.species_label import SpeciesLabel

    return SpeciesLabel


async def _put(session, label):
    from fishsense_api.controllers.label_controller import put_species_label

    row_id = await put_species_label(11, label, session=session)
    await session.flush()
    return row_id


async def _row(session):
    rows = (await session.exec(select(_model()))).all()
    assert len(rows) == 1
    return rows[0]


async def test_angle_degrees_round_trips(session):
    """A reviewed angle written through the PUT survives the round trip."""
    model = _model()
    await _put(
        session,
        model(
            id=None,
            label_studio_task_id=500,
            label_studio_project_id=284573,
            fish_angle_degrees=20.0,
        ),
    )
    assert (await _row(session)).fish_angle_degrees == 20.0


async def test_unreviewed_label_has_no_angle(session):
    """Absent means unreviewed, not zero. 0° is a real value in the sweep, so
    a default of 0.0 would be indistinguishable from a genuine answer."""
    model = _model()
    await _put(
        session,
        model(id=None, label_studio_task_id=501, label_studio_project_id=284573),
    )
    assert (await _row(session)).fish_angle_degrees is None


async def test_zero_degrees_is_stored_and_is_not_null(session):
    """0.0 must persist as 0.0. A falsy check anywhere in the write path would
    collapse the first step of every sweep back into 'unreviewed'."""
    model = _model()
    await _put(
        session,
        model(
            id=None,
            label_studio_task_id=502,
            label_studio_project_id=284573,
            fish_angle_degrees=0.0,
        ),
    )
    row = await _row(session)
    assert row.fish_angle_degrees == 0.0
    assert row.fish_angle_degrees is not None


async def test_a_put_that_omits_the_angle_preserves_it(session):
    """The shape of the hourly species sync: read a label, mutate the fields it
    owns, PUT the whole model back. A writer that predates this column — an
    older SDK, or the sync itself — must not blank a reviewed angle.

    This is the same guarantee `test_label_put_preserves_unmentioned_fields`
    establishes generally; asserted again here because the value it protects is
    hand-reviewed and has no other source to recover from.
    """
    model = _model()
    await _put(
        session,
        model(
            id=None,
            label_studio_task_id=503,
            label_studio_project_id=284573,
            fish_angle_degrees=35.0,
        ),
    )

    # The sync's write: same natural key, category set, angle never mentioned.
    await _put(
        session,
        model(
            id=None,
            label_studio_task_id=503,
            label_studio_project_id=284573,
            fish_angle_category="x > 15°",
            completed=True,
        ),
    )

    row = await _row(session)
    assert row.fish_angle_degrees == 35.0, "sync must not clear a reviewed angle"
    assert row.fish_angle_category == "x > 15°"
    assert row.completed is True

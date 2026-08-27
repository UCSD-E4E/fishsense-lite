"""`needs_reprocess` marks a label whose overlay JPEG is stale.

The processed JPEG a labeler sees is not regenerated once it exists: every
preprocess cohort selects on "image has no label row of this kind", so the
moment populate seeds a row the image drops out and its JPEG is frozen. That
is the right default -- rectifying a dive's raw `.ORF`s is hours of CPU -- but
it means a change to what the overlay *draws* reaches only images that have
not been preprocessed yet.

Stage 0.1's expected-laser box moved on 2026-08-27 (it was clipping real
lasers, and two dives had their median laser outside it entirely), which is
the first time that mattered. `needs_reprocess` is the way to say "this
label's image must be drawn again" without deleting the label and losing the
work a labeler already did.

One flag per label kind rather than one on `Image`, because an image carries a
different JPEG per stage -- `preprocess_jpeg`, `preprocess_groups_jpeg`,
`preprocess_headtail_jpeg`, `preprocess_slate_images_jpeg` -- and a change to
one stage's overlay says nothing about the other three.

Nothing in the pipeline *reads* the flag yet: making a cohort honour it is a
separate, deliberate change, since it re-stages raw bytes from the NAS and
re-runs per-image work for every flagged row.
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


def _label_kinds():
    from fishsense_api.models.dive_slate_label import DiveSlateLabel
    from fishsense_api.models.head_tail_label import HeadTailLabel
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.species_label import SpeciesLabel

    return [
        pytest.param(LaserLabel, id="laser"),
        pytest.param(SpeciesLabel, id="species"),
        pytest.param(HeadTailLabel, id="headtail"),
        pytest.param(DiveSlateLabel, id="dive-slate"),
    ]


@pytest.mark.parametrize("model", _label_kinds())
def test_flag_defaults_to_false_not_none(model):
    """False, not NULL: the flag is read as a predicate, and a three-valued
    one turns `WHERE NOT needs_reprocess` into a silent row filter -- the same
    shape as the `created_at IS NULL` bug that made `ORDER BY created_at DESC`
    match nothing on `laserextrinsics`.
    """
    assert model(image_id=1).needs_reprocess is False


@pytest.mark.parametrize("model", _label_kinds())
async def test_flag_persists_and_can_be_cleared(session, model):
    session.add(model(image_id=7, label_studio_project_id=1, needs_reprocess=True))
    await session.commit()

    row = (
        await session.exec(select(model).where(model.image_id == 7))
    ).one()
    assert row.needs_reprocess is True

    row.needs_reprocess = False
    session.add(row)
    await session.commit()
    refreshed = (
        await session.exec(select(model).where(model.image_id == 7))
    ).one()
    assert refreshed.needs_reprocess is False


@pytest.mark.parametrize("model", _label_kinds())
async def test_flag_is_queryable_as_a_cohort(session, model):
    """The point of the column: select the stale rows without touching the
    label payload."""
    session.add(model(image_id=1, label_studio_project_id=1, needs_reprocess=True))
    session.add(model(image_id=2, label_studio_project_id=1, needs_reprocess=False))
    session.add(model(image_id=3, label_studio_project_id=1))
    await session.commit()

    stale = (
        await session.exec(select(model).where(model.needs_reprocess.is_(True)))
    ).all()
    assert [r.image_id for r in stale] == [1]


@pytest.mark.parametrize("model", _label_kinds())
def test_flag_is_independent_of_superseded(model):
    """`superseded` dead-letters a *label*; `needs_reprocess` marks its
    *image*. A perfectly good label can need a redraw, and a superseded one
    still occupies the row that keeps its image out of the cohort."""
    label = model(image_id=1, superseded=False, needs_reprocess=True)
    assert (label.superseded, label.needs_reprocess) == (False, True)

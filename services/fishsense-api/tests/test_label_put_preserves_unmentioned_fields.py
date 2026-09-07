"""A PUT must not clear fields the caller never mentioned.

The four `put_*_label` handlers rebuilt a full SQLModel from the request body
and `session.merge`d the whole thing. Anything absent from the body took its
model default -- so a writer that constructs a label with the twelve fields it
cares about silently wiped the ones it did not.

That is not hypothetical, and it is not new. CLAUDE.md already records it for
`LaserPrediction`: "the persist activity builds LaserPrediction without the
gate fields and the upsert merges the whole model, so re-predicting a dive
clears its verdicts." Same defect, different table, and nothing generalised the
lesson.

Prod, 2026-09-07: `populate_laser_label_studio_project_activity._record` builds
a fresh `LaserLabel` with thirteen explicit kwargs and no `needs_reprocess`,
and its `_select_unlabeled_images` selects images with no *completed* label --
which is exactly the set a reprocess flag marks. Populate runs hourly, so a
laser reprocess flag had a useful life of under an hour: dive 442's 259 flags
were set at 22:05 and gone before the render they requested had started.

The fix is at the API rather than in each caller because "remember to restate
every field you do not want cleared" has now been got wrong twice, in two
tables, and the next writer would have to remember it a third time.
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


def _kinds():
    from fishsense_api.controllers import label_controller as lc
    from fishsense_api.models.dive_slate_label import DiveSlateLabel
    from fishsense_api.models.head_tail_label import HeadTailLabel
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.species_label import SpeciesLabel

    return [
        pytest.param(LaserLabel, lc.put_laser_label, id="laser"),
        pytest.param(SpeciesLabel, lc.put_species_label, id="species"),
        pytest.param(HeadTailLabel, lc.put_headtail_label, id="headtail"),
        pytest.param(DiveSlateLabel, lc.put_dive_slate_label, id="dive-slate"),
    ]


async def _seed_image(session, image_id=11):
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 9, 7, tzinfo=timezone.utc)
    session.add(Dive(id=1, path="/d1", priority=Priority.HIGH, dive_datetime=when))
    session.add(
        Image(
            id=image_id,
            dive_id=1,
            path=f"/i{image_id}.ORF",
            checksum=f"{image_id:032d}",
            is_canonical=True,
            taken_datetime=when,
        )
    )
    await session.flush()


@pytest.mark.parametrize("model,handler", _kinds())
class TestUnmentionedFieldsSurvive:
    async def test_a_put_that_omits_needs_reprocess_does_not_clear_it(
        self, session, model, handler
    ):
        """The prod case: populate rewrites the row and the flag disappears."""
        await _seed_image(session)
        session.add(
            model(
                image_id=11,
                label_studio_project_id=7,
                completed=False,
                needs_reprocess=True,
            )
        )
        await session.flush()

        # what populate sends: explicit kwargs, no `needs_reprocess`
        payload = model(
            image_id=11,
            label_studio_project_id=7,
            label_studio_task_id=99,
            completed=False,
        )
        await handler(11, payload, session=session)
        await session.flush()

        rows = (await session.exec(select(model).where(model.image_id == 11))).all()
        assert len(rows) == 1, "natural-key upsert must not append a row"
        assert rows[0].needs_reprocess is True, (
            "a field the caller never mentioned must survive the write"
        )
        assert rows[0].label_studio_task_id == 99, "provided fields still apply"

    async def test_a_put_that_explicitly_sends_false_does_clear_it(
        self, session, model, handler
    ):
        """Preserving the unmentioned must not make the field unwritable."""
        await _seed_image(session)
        session.add(
            model(image_id=11, label_studio_project_id=7, needs_reprocess=True)
        )
        await session.flush()

        payload = model(
            image_id=11, label_studio_project_id=7, needs_reprocess=False
        )
        await handler(11, payload, session=session)
        await session.flush()

        rows = (await session.exec(select(model).where(model.image_id == 11))).all()
        assert rows[0].needs_reprocess is False

    async def test_a_new_row_is_still_created(self, session, model, handler):
        await _seed_image(session)
        payload = model(image_id=11, label_studio_project_id=7, completed=True)
        await handler(11, payload, session=session)
        await session.flush()

        rows = (await session.exec(select(model).where(model.image_id == 11))).all()
        assert len(rows) == 1
        assert rows[0].completed is True

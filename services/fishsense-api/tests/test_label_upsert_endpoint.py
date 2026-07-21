"""put_*_label endpoints upsert on the natural key, not blind-INSERT.

Every label model carries a `uq_<kind>_image_project` unique constraint on
`(image_id, label_studio_project_id)`. The handlers used a plain
`session.merge(label)` which keys on the primary key alone, so a body with
`id=None` always INSERTs. That is fine on the first write but blows up with
a duplicate-key 500 on the *second* — exactly what happens when the populate
activity retries and re-writes the sentinel rows it already created (observed
in prod on headtail dive 347 after the LS import dedup fix stopped masking it
by crashing earlier).

Same failure mode and same fix as `post_measurement` /
`uq_measurement_image_fish`: resolve the natural key to the existing row id
first so the merge becomes an UPDATE. Parametrized across all four label
kinds because all four handlers shared the bug.
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


def _cases():
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        put_dive_slate_label,
        put_headtail_label,
        put_laser_label,
        put_species_label,
    )
    from fishsense_api.models.dive_slate_label import (  # pylint: disable=import-outside-toplevel
        DiveSlateLabel,
    )
    from fishsense_api.models.head_tail_label import (  # pylint: disable=import-outside-toplevel
        HeadTailLabel,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )
    from fishsense_api.models.species_label import (  # pylint: disable=import-outside-toplevel
        SpeciesLabel,
    )

    return [
        pytest.param(put_headtail_label, HeadTailLabel, id="headtail"),
        pytest.param(put_laser_label, LaserLabel, id="laser"),
        pytest.param(put_species_label, SpeciesLabel, id="species"),
        pytest.param(put_dive_slate_label, DiveSlateLabel, id="dive_slate"),
    ]


def _label(model, *, task_id, project_id, completed):
    return model(
        id=None,
        label_studio_task_id=task_id,
        label_studio_project_id=project_id,
        completed=completed,
    )


@pytest.mark.parametrize("put_fn,model", _cases())
async def test_put_label_creates_then_upserts_on_retry(session, put_fn, model):
    """First PUT inserts; a second PUT for the same (image, project) with
    id=None must UPDATE the same row — not raise duplicate-key."""
    first_id = await put_fn(
        11, _label(model, task_id=500, project_id=73, completed=False), session=session
    )
    await session.flush()

    # The populate retry: same task, same (image, project), still id=None.
    second_id = await put_fn(
        11, _label(model, task_id=500, project_id=73, completed=True), session=session
    )
    await session.flush()

    rows = (await session.exec(select(model))).all()
    assert len(rows) == 1, "retry must upsert on (image_id, project), not duplicate"
    assert first_id == second_id
    assert rows[0].completed is True, "latest write wins"


@pytest.mark.parametrize("put_fn,model", _cases())
async def test_put_label_distinct_projects_same_image_both_persist(
    session, put_fn, model
):
    """The natural key includes the project — the same image in two
    different per-dive projects is two legitimate rows."""
    await put_fn(
        11, _label(model, task_id=600, project_id=73, completed=False), session=session
    )
    await session.flush()
    await put_fn(
        11, _label(model, task_id=601, project_id=99, completed=False), session=session
    )
    await session.flush()

    rows = (await session.exec(select(model))).all()
    assert len(rows) == 2

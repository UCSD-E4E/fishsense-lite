"""Known-size reference for the physical fish models + the accuracy view.

The models (and the ruler) are the pipeline's held-out VALIDATION set: their
true lengths are never fed into calibration (that would be circular), only
compared against what stage 14 measures. Persisting them here turns that
comparison into a queryable artifact instead of a number in someone's notes,
and backs a Superset accuracy dashboard.

`fish_model_measurement_accuracy` joins measurements to the reference through
`Fish.name` — the same natural key stage 14 resolves model identity by — so a
new measurement shows up in the view with no extra wiring.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.views import (
    FISH_MODEL_ACCURACY_VIEW_SQL,
    KNOWN_FISH_MODELS,
)


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(text(FISH_MODEL_ACCURACY_VIEW_SQL))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


# ── the seed data itself ──────────────────────────────────────────────


def test_known_models_cover_the_labeled_taxonomy():
    """Every `Fish Model, <name>` leaf a labeler can pick must have a known
    length, or its measurements can never be graded."""
    names = {m["name"] for m in KNOWN_FISH_MODELS}
    assert {"Snook", "Grouper", "Shark", "Purple Angel"} <= names


def test_known_lengths_are_positive_and_plausible():
    for model in KNOWN_FISH_MODELS:
        assert 0.05 < model["known_length_m"] < 2.0, model


def test_known_model_names_are_unique():
    names = [m["name"] for m in KNOWN_FISH_MODELS]
    assert len(names) == len(set(names))


# ── the reference table ───────────────────────────────────────────────


async def test_reference_row_round_trips(session):
    from fishsense_api.models.fish_model_reference import (  # pylint: disable=import-outside-toplevel
        FishModelReference,
    )

    session.add(FishModelReference(name="Grouper", known_length_m=0.360))
    await session.flush()

    row = (
        await session.exec(
            select(FishModelReference).where(FishModelReference.name == "Grouper")
        )
    ).first()
    assert row is not None
    assert row.known_length_m == pytest.approx(0.360)


# ── the accuracy view ─────────────────────────────────────────────────


async def _seed_measurement(session, *, dive_id, image_id, model_name, length_m):
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    existing = (await session.exec(select(Dive).where(Dive.id == dive_id))).first()
    if existing is None:
        session.add(
            Dive(
                id=dive_id,
                path=f"/dev/null/{dive_id}",
                dive_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
                priority=Priority.HIGH,
            )
        )
        await session.flush()
    fish = (
        await session.exec(select(Fish).where(Fish.name == model_name))
    ).first()
    if fish is None:
        fish = Fish(name=model_name, species_id=None)
        session.add(fish)
        await session.flush()
    session.add(
        Image(
            id=image_id,
            path=f"/dev/null/img-{image_id}",
            taken_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            checksum=f"img-{image_id:032d}"[:32],
            dive_id=dive_id,
        )
    )
    await session.flush()
    session.add(Measurement(image_id=image_id, fish_id=fish.id, length_m=length_m))
    await session.flush()


async def _rows(session):
    result = await session.exec(
        text(
            "SELECT dive_id, model_name, known_length_m, length_m, "
            "error_m, pct_error FROM fish_model_measurement_accuracy "
            "ORDER BY image_id"
        )
    )
    return [dict(r._mapping) for r in result]  # pylint: disable=protected-access


async def test_accuracy_view_computes_error_against_known_length(session):
    from fishsense_api.models.fish_model_reference import (  # pylint: disable=import-outside-toplevel
        FishModelReference,
    )

    session.add(FishModelReference(name="Grouper", known_length_m=0.360))
    await session.flush()
    # Measured 10% long.
    await _seed_measurement(
        session, dive_id=1, image_id=11, model_name="Grouper", length_m=0.396
    )

    rows = await _rows(session)

    assert len(rows) == 1
    assert rows[0]["model_name"] == "Grouper"
    assert rows[0]["known_length_m"] == pytest.approx(0.360)
    assert rows[0]["error_m"] == pytest.approx(0.036, abs=1e-6)
    assert rows[0]["pct_error"] == pytest.approx(10.0, abs=1e-4)


async def test_accuracy_view_excludes_real_fish(session):
    """Real (wild) fish carry name=NULL and have no reference row — they must
    not appear, or the view stops being a model-accuracy view."""
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.fish_model_reference import (  # pylint: disable=import-outside-toplevel
        FishModelReference,
    )

    session.add(FishModelReference(name="Grouper", known_length_m=0.360))
    session.add(
        Dive(
            id=1,
            path="/dev/null/1",
            dive_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            priority=Priority.HIGH,
        )
    )
    await session.flush()
    wild = Fish(name=None, species_id=None)
    session.add(wild)
    await session.flush()
    session.add(
        Image(
            id=11,
            path="/dev/null/img-11",
            taken_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            checksum="c" * 32,
            dive_id=1,
        )
    )
    await session.flush()
    session.add(Measurement(image_id=11, fish_id=wild.id, length_m=0.5))
    await session.flush()

    assert await _rows(session) == []


async def test_accuracy_view_excludes_models_without_a_reference(session):
    """A model nobody has measured with calipers yet can't be graded — it must
    be absent rather than silently compared against NULL."""
    await _seed_measurement(
        session, dive_id=1, image_id=11, model_name="Unmeasured Model", length_m=0.4
    )

    assert await _rows(session) == []

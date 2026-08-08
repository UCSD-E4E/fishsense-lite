"""Species-mislabel detection for the physical fish models.

A model's measured length is a strong prior on *which* model it is, so a frame
whose length matches a different model far better than its own label is a
mislabel suspect. Found 18 real ones in prod on 2026-08-04 (16 in contiguous
runs — a whole photo sequence of one model labeled as another).

The asymmetry this view exists to handle: stage 14 measures the *projection*,
so an out-of-plane fish reads SHORT and never long. Hence over-measurement is
`high` (geometry cannot explain it) while "short and matches another model" is
`medium` — a Snook (455mm) angled ~21 deg reads 360mm, exactly Grouper.

A group-maximum signal was tried and removed after it failed on real data; the
last two tests here are the regressions that killed it.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.views import (
    FISH_MODEL_ACCURACY_VIEW_SQL,
    FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
)

# Known lengths used by every test here (mirrors prod).
_KNOWN = {
    "Snook": 0.455,
    "Grouper": 0.360,
    "Shark": 0.605,
    "Purple Angel": 0.192,
    "Gray Anthias": 0.195,
    "Yellow Anthias": 0.200,
}


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(text(FISH_MODEL_ACCURACY_VIEW_SQL))
        await conn.execute(text(FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        await _seed_reference(s)
        yield s
    await engine.dispose()


async def _seed_reference(session):
    from fishsense_api.models.fish_model_reference import (  # pylint: disable=import-outside-toplevel
        FishModelReference,
    )

    for name, length in _KNOWN.items():
        session.add(FishModelReference(name=name, known_length_m=length))
    await session.flush()


_NEXT = {"image": 1000}


async def _measure(session, *, dive_id, model_name, length_m):
    """Seed one measurement of `model_name` at `length_m` in `dive_id`."""
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel
    from sqlmodel import select  # pylint: disable=import-outside-toplevel

    if (await session.exec(select(Dive).where(Dive.id == dive_id))).first() is None:
        session.add(
            Dive(
                id=dive_id,
                path=f"/dev/null/{dive_id}",
                dive_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
                priority=Priority.HIGH,
            )
        )
        await session.flush()
    fish = (await session.exec(select(Fish).where(Fish.name == model_name))).first()
    if fish is None:
        fish = Fish(name=model_name, species_id=None)
        session.add(fish)
        await session.flush()
    _NEXT["image"] += 1
    image_id = _NEXT["image"]
    session.add(
        Image(
            id=image_id,
            path=f"/dev/null/img-{image_id}",
            taken_datetime=datetime(2023, 8, 29, tzinfo=timezone.utc),
            checksum=f"{image_id:032d}",
            dive_id=dive_id,
        )
    )
    await session.flush()
    session.add(Measurement(image_id=image_id, fish_id=fish.id, length_m=length_m))
    await session.flush()
    return image_id


async def _suspects(session):
    result = await session.exec(
        text(
            "SELECT image_id, dive_id, labeled_model, best_fit_model, "
            "confidence FROM fish_model_species_mislabel_suspects "
            "ORDER BY image_id"
        )
    )
    return [dict(r._mapping) for r in result]  # pylint: disable=protected-access


# ── clean data ────────────────────────────────────────────────────────


async def test_correctly_labeled_models_are_not_flagged(session):
    for name, length in _KNOWN.items():
        await _measure(session, dive_id=1, model_name=name, length_m=length)

    assert await _suspects(session) == []


async def test_mild_foreshortening_is_not_flagged(session):
    """Frames a few percent short are normal projection loss, not mislabels."""
    for pct in (0.0, -0.02, -0.05, -0.08):
        await _measure(
            session, dive_id=1, model_name="Snook", length_m=0.455 * (1 + pct)
        )

    assert await _suspects(session) == []


# ── high-confidence signal: over-measurement ──────────────────────────


async def test_over_measured_frame_is_high_confidence(session):
    """Foreshortening cannot lengthen a fish, so a frame measuring far LONGER
    than its label is a mislabel regardless of geometry. (Prod: dive 84 image
    4868, labeled Purple Angel, measured 351mm = Grouper, +83%.)"""
    await _measure(session, dive_id=1, model_name="Purple Angel", length_m=0.351)

    rows = await _suspects(session)

    assert len(rows) == 1
    assert rows[0]["labeled_model"] == "Purple Angel"
    assert rows[0]["best_fit_model"] == "Grouper"
    assert rows[0]["confidence"] == "high"


# ── the ambiguous case foreshortening creates ─────────────────────────


async def test_short_frame_matching_another_model_is_medium(session):
    """A Snook reading 330mm fits Grouper — but a Snook angled ~21 deg reads
    exactly that, so length alone cannot decide. Surface for review, don't
    accuse. (Prod: the 16 Snook->Grouper frames, later confirmed real.)"""
    await _measure(session, dive_id=1, model_name="Snook", length_m=0.330)

    rows = await _suspects(session)

    assert len(rows) == 1
    assert rows[0]["best_fit_model"] == "Grouper"
    assert rows[0]["confidence"] == "medium"


# ── real-data regressions: a group-max signal was tried and removed ───


async def test_near_identical_lengths_do_not_produce_flags(session):
    """Purple Angel 0.192 / Gray Anthias 0.195 / Yellow Anthias 0.200 are
    within 4%, so length cannot discriminate them. Correctly-labeled frames
    that happen to sit nearest a neighbour's length must NOT be flagged.

    Regression: an earlier group-maximum rule condemned 15 correct dive-59
    Purple Angels because the group's max landed nearer Yellow Anthias.
    """
    for length in (0.192, 0.196, 0.199, 0.200):
        await _measure(session, dive_id=1, model_name="Purple Angel", length_m=length)

    assert await _suspects(session) == []


async def test_one_bad_frame_does_not_condemn_its_neighbours(session):
    """A single mislabeled frame must not drag correctly-labeled frames of the
    same model into the suspect list.

    Regression: the group-maximum rule let prod dive 60's single 601mm frame
    inflate the Grouper group's max, flagging all 19 correct Groupers as Shark.
    """
    for _ in range(6):
        await _measure(session, dive_id=1, model_name="Grouper", length_m=0.355)
    bad = await _measure(session, dive_id=1, model_name="Grouper", length_m=0.601)

    rows = await _suspects(session)

    assert [r["image_id"] for r in rows] == [bad], "only the bad frame"
    assert rows[0]["best_fit_model"] == "Shark"
    assert rows[0]["confidence"] == "high"

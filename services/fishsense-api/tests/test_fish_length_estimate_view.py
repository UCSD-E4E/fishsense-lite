"""Per-fish length estimate — p90 over a fish's frames, not the mean.

Stage 14 back-projects head and tail at ONE laser-derived depth, so it measures
the fish's *projection*: any out-of-plane angle can only SHORTEN it. Per-frame
error is therefore one-sided-negative (measured skew -4.87 over 437 fish-model
frames), which makes the mean a biased-low estimator and a high quantile the
right one. Switching mean -> p90 halved absolute error across 23 dive x model
groups (4.35% -> 2.26%) with no change to the measurements themselves.

These tests pin that property, so a well-meaning "just average the frames"
refactor fails loudly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.views import FISH_LENGTH_ESTIMATE_VIEW_SQL


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=unused-import

    _SEEDED_DIVES.clear()
    _SEEDED_FISH.clear()
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(text(FISH_LENGTH_ESTIMATE_VIEW_SQL))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


_SEEDED_DIVES: set[int] = set()
_SEEDED_FISH: set[int] = set()


def _seed(session, fish_id: int, dive_id: int, lengths: list[float], base: int = 0):
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.fish import Fish
    from fishsense_api.models.image import Image
    from fishsense_api.models.measurement import Measurement
    from fishsense_api.models.priority import Priority

    if dive_id not in _SEEDED_DIVES:
        _SEEDED_DIVES.add(dive_id)
        session.add(
            Dive(
                id=dive_id,
                path=f"/dev/null/{dive_id}",
                dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
                priority=Priority["HIGH"],
            )
        )
    if fish_id not in _SEEDED_FISH:
        _SEEDED_FISH.add(fish_id)
        session.add(Fish(id=fish_id, name=f"model{fish_id}"))
    for k, ln in enumerate(lengths):
        iid = base + k + 1
        session.add(
            Image(
                id=iid,
                path=f"/dev/null/img-{iid}",
                taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
                checksum=f"img-{iid:032d}"[:32],
                dive_id=dive_id,
            )
        )
        session.add(Measurement(image_id=iid, fish_id=fish_id, length_m=ln))


async def _row(session, fish_id: int, dive_id: int):
    res = await session.exec(
        text(
            "SELECT * FROM fish_length_estimate "
            "WHERE fish_id = :f AND dive_id = :d"
        ).bindparams(f=fish_id, d=dive_id)
    )
    row = res.mappings().first()
    return dict(row) if row is not None else None


async def test_p90_rejects_the_one_sided_foreshortening_tail(session):
    """THE motivating case. Nine good frames plus one badly foreshortened one:
    the mean is dragged down by the tail, p90 is not."""
    _seed(session, 1, 1, [1.00] * 9 + [0.50])
    await session.flush()

    row = await _row(session, 1, 1)
    assert row["n_frames"] == 10
    assert row["length_p90_m"] == pytest.approx(1.00)
    assert row["length_mean_m"] == pytest.approx(0.95)
    assert row["length_min_m"] == pytest.approx(0.50), "the bad frame is still there"


async def test_p90_is_not_simply_the_max_when_there_are_enough_frames(session):
    """p90 must reject the top frame's label noise too, or it is just `max`
    under another name. 20 frames -> rank ceil(0.9*20) = 18, not 20."""
    _seed(session, 1, 1, [0.90] * 17 + [1.00, 1.10, 1.20])
    await session.flush()

    row = await _row(session, 1, 1)
    assert row["n_frames"] == 20
    assert row["length_max_m"] == pytest.approx(1.20)
    assert row["length_p90_m"] == pytest.approx(1.00), "rank 18 of 20"


async def test_p90_degenerates_to_max_for_small_n(session):
    """Nearest-rank p90 on n<=8 IS the max — inherent, and why `n_frames` is
    exposed so consumers can filter. Pinned so it is a known property, not a
    surprise."""
    _seed(session, 1, 1, [0.90, 0.95, 1.00])
    await session.flush()

    row = await _row(session, 1, 1)
    assert row["n_frames"] == 3
    assert row["length_p90_m"] == pytest.approx(1.00) == row["length_max_m"]


async def test_median_is_the_nearest_rank_middle(session):
    _seed(session, 1, 1, [0.80, 0.90, 1.00, 1.10, 1.20])
    await session.flush()

    row = await _row(session, 1, 1)
    assert row["length_median_m"] == pytest.approx(1.00)


async def test_one_row_per_fish_and_dive(session):
    """A model Fish is global (one per model across dives), so grouping must
    include dive_id or two dives' frames would be pooled into one estimate."""
    _seed(session, 1, 1, [1.00, 1.00, 1.00], base=0)
    _seed(session, 1, 2, [2.00, 2.00, 2.00], base=100)
    await session.flush()

    assert (await _row(session, 1, 1))["length_p90_m"] == pytest.approx(1.00)
    assert (await _row(session, 1, 2))["length_p90_m"] == pytest.approx(2.00)


async def test_measurements_without_a_length_are_excluded(session):
    _seed(session, 1, 1, [1.00, 1.00])
    from fishsense_api.models.image import Image
    from fishsense_api.models.measurement import Measurement

    session.add(
        Image(
            id=900,
            path="/dev/null/img-900",
            taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
            checksum="img-900",
            dive_id=1,
        )
    )
    session.add(Measurement(image_id=900, fish_id=1, length_m=None))
    await session.flush()

    assert (await _row(session, 1, 1))["n_frames"] == 2

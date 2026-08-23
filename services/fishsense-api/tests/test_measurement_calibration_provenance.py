# pylint: disable=C0121
"""Which calibration a `Measurement` was computed with.

A fish length is only meaningful relative to the `LaserExtrinsics` that
produced the depth behind it, and calibrations do get replaced — the
2026-08-11 slate panel-offset fix recalibrated 6 of the 8 dives that already
had measurements. Before this column there was no way to ask "is this length
still current?", so the choice was to rewrite every measurement on every run
or to leave stale ones in place forever. Stamping the provenance turns it
into an ordinary drainable cohort: stage 14 skips an image only when its
measurement was computed with the calibration that would be used *today*, and
re-measures it otherwise.

Legacy rows carry NULL and therefore read as stale exactly once — that is the
length half of the depth backfill, and it converges without a separate script.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from tests_support.stage14_fixtures import (
    fish_model_measurable_image as _measurable_image,
    measurement,
)


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _dive(dive_id: int, *, calibration_dive_id: int | None = None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority.HIGH,
        calibration_dive_id=calibration_dive_id,
    )


def _extrinsics(extrinsics_id: int, dive_id: int):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    return LaserExtrinsics(
        id=extrinsics_id,
        laser_position=[0.1, 0.0, 0.0],
        laser_axis=[0.0, 0.0, 1.0],
        dive_id=dive_id,
        camera_id=1,
    )


def _measurement(image_id: int, *, laser_extrinsics_id=None, fish_id: int = 100):
    return measurement(
        image_id, fish_id=fish_id, laser_extrinsics_id=laser_extrinsics_id
    )


async def test_post_measurement_persists_the_calibration_it_used(session):
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        post_measurement,
    )
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    await session.flush()

    measurement_id = await post_measurement(
        100, _measurement(11, laser_extrinsics_id=51), session=session
    )

    row = await session.get(Measurement, measurement_id)
    assert row.laser_extrinsics_id == 51


async def test_post_measurement_updates_provenance_on_re_measure(session):
    """The upsert keys on `(image_id, fish_id)`, so a re-measure after a
    recalibration must move the provenance forward with the length — a row
    still claiming the old calibration would be re-selected forever."""
    from fishsense_api.controllers.fish_controller import (  # pylint: disable=import-outside-toplevel
        post_measurement,
    )
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    await session.flush()

    first = await post_measurement(
        100, _measurement(11, laser_extrinsics_id=50), session=session
    )
    second = await post_measurement(
        100, _measurement(11, laser_extrinsics_id=51), session=session
    )

    assert first == second
    row = await session.get(Measurement, second)
    assert row.laser_extrinsics_id == 51


async def test_cohort_skips_a_dive_measured_with_the_current_calibration(session):
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=51))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None


async def test_cohort_repicks_a_dive_after_recalibration(session):
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=50))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 1


async def test_cohort_repicks_a_dive_whose_measurements_predate_provenance(session):
    """Every measurement in prod today carries NULL here. They re-enter the
    cohort once, get recomputed under the current calibration, and drain."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _extrinsics(51, 1)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=None))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 1


async def test_cohort_resolves_provenance_through_a_borrowed_calibration(session):
    """A fish-only dive is measured with the sibling's extrinsics row, so
    that is the id its measurements must carry — not its own (it has none)."""
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1, calibration_dive_id=2), _dive(2), _extrinsics(51, 2)])
    _measurable_image(session, 11, 1)
    session.add(_measurement(11, laser_extrinsics_id=51))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None


# ── the correlation bug ───────────────────────────────────────────────
#
# `_resolved_laser_extrinsics_id()` is a scalar subquery used deep inside a
# NOT EXISTS. SQLAlchemy only auto-correlates against the *immediately*
# enclosing SELECT, so without an explicit `.correlate(Dive)` it emitted
# `FROM laserextrinsics, dive` — an uncorrelated cross join returning one row
# per extrinsics row in the table.
#
# Postgres rejects that outright (CardinalityViolationError: "more than one
# row returned by a subquery used as an expression"), which took both cohort
# selectors down in prod: every hourly poll 500'd, stage 14 stamped no
# provenance, and the laser-depth stage drained exactly one dive before
# stopping. SQLite silently returns the first row instead of raising, and
# every fixture here seeded a single extrinsics row, so the wrong SQL and the
# right SQL agreed and the suite stayed green.
#
# These two tests close that gap from both directions: one seeds a second
# extrinsics row so the uncorrelated form picks the *wrong* dive's
# calibration even on SQLite, and one asserts the emitted SQL is correlated
# regardless of dialect.


async def test_cohort_resolves_each_dives_own_calibration_not_the_first_row(session):
    """Two calibrated dives, and the one under test is NOT the lowest id.

    Uncorrelated, `coalesce(...)` returns extrinsics 51 (dive 1's, the first
    row in the table) for every dive, so dive 2's correctly-stamped
    measurement looks stale and the dive is re-selected forever.
    """
    session.add_all([_dive(1), _dive(2), _extrinsics(51, 1), _extrinsics(52, 2)])
    _measurable_image(session, 21, 2)
    session.add(_measurement(21, laser_extrinsics_id=52))
    await session.flush()

    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    assert await select_next_for_measure_fish(session=session) is None


def test_resolved_extrinsics_subquery_is_correlated():
    """Dialect-independent guard on the shape of the shipped SQL.

    Compiles the real cohort query — the nesting matters, since the bug only
    appears when the scalar subquery sits inside a NOT EXISTS inside an
    EXISTS. The subquery must *reference* the outer `dive`, never select from
    it. This is the assertion that would have caught the prod outage without
    a Postgres to run against.
    """
    import re  # pylint: disable=import-outside-toplevel

    from sqlalchemy.dialects import postgresql  # pylint: disable=import-outside-toplevel

    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        _laser_depth_cohort_query,
    )

    compiled = str(_laser_depth_cohort_query().compile(dialect=postgresql.dialect()))

    assert "FROM laserextrinsics, dive" not in compiled, (
        "scalar subquery is uncorrelated — it cross-joins dive and returns one "
        f"row per extrinsics row, which Postgres rejects:\n{compiled}"
    )

    # Pin the coalesce's own subqueries, not just the absence of one string.
    # `laserextrinsics.dive_id = dive.id` also appears in the unrelated
    # `has_laser_extrinsics` EXISTS, so asserting on it alone passed happily
    # while the scalar subquery was still cross-joining.
    start = compiled.index("coalesce(") + len("coalesce")
    depth = 0
    end = start
    for end, char in enumerate(compiled[start:], start):
        depth += (char == "(") - (char == ")")
        if depth == 0:
            break
    else:  # pragma: no cover - unbalanced parens would mean a broken compiler
        raise AssertionError(f"unbalanced parentheses in compiled SQL:\n{compiled}")
    inside_coalesce = compiled[start : end + 1]

    for from_clause in re.findall(r"FROM (\w+(?:, \w+)*)", inside_coalesce):
        assert from_clause == "laserextrinsics", (
            "a subquery inside coalesce() selects from more than "
            f"laserextrinsics ({from_clause!r}) — it is not correlated:\n{compiled}"
        )

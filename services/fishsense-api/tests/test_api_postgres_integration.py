"""fishsense-api against a real Postgres — the startup path a deploy takes.

Every other test in this package runs on in-memory SQLite, which means the
`lifespan` sequence has never been exercised against the database it actually
runs on. That sequence *is* the deploy: `create_all`, then
`run_alembic_upgrade`. It has taken the API down before (the `laserprediction`
duplicate-table crash-loop), and `runtime-import` only imports the module — it
never starts it.

Two things only Postgres can tell us:

  * whether the four views in `views.py` exist after startup. They are raw SQL
    in migrations, not `SQLModel.metadata`, so `create_all` cannot produce
    them.
  * whether their SQL is even valid Postgres. `DISTINCT ON`, `FILTER (WHERE
    ...)`, `PERCENTILE_CONT` and friends parse differently or not at all on
    SQLite, so a broken view can pass the whole unit suite.

Run with the local stack up (`docker compose -f deploy/compose.local.yml up -d
postgres`) via `./check.sh integration`, or point `FISHSENSE_POSTGRES_*` at any
Postgres 17.

The module creates and drops its own scratch database and never touches
`fishsense`.
"""

from __future__ import annotations

import asyncio
import os

import pytest

_PG_HOST = os.environ.get("FISHSENSE_POSTGRES_HOST", "postgres")
_PG_PORT = os.environ.get("FISHSENSE_POSTGRES_PORT", "5432")
_PG_USER = os.environ.get("FISHSENSE_POSTGRES_USER", "postgres")
_PG_PASSWORD = os.environ.get("FISHSENSE_POSTGRES_PASSWORD", "fishsense_local")
_SCRATCH_DB = os.environ.get("FISHSENSE_POSTGRES_ITEST_DB", "fishsense_api_itest")

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _point_settings_at_the_scratch_database():
    """Repoint dynaconf at the scratch DB, but only while these tests run.

    Emphatically NOT module-level `os.environ[...] = ...`: pytest imports every
    test module during collection, including on a default `-m 'not
    integration'` run, so that would overwrite the session's Postgres settings
    for suites that never asked for it — and the sibling modules' deliberate
    `os.environ.setdefault(..., "ignored")` would then silently no-op, pointing
    them at a real database.

    Dynaconf caches on first attribute access, so setting the environment isn't
    enough on its own; `reload()` re-runs the loaders. Both are undone
    afterwards so the rest of the session sees what it did before.
    """
    from _pytest.monkeypatch import MonkeyPatch  # pylint: disable=import-outside-toplevel

    from fishsense_api.config import settings  # pylint: disable=import-outside-toplevel

    monkeypatch = MonkeyPatch()
    monkeypatch.setenv("E4EFS_POSTGRES__HOST", _PG_HOST)
    monkeypatch.setenv("E4EFS_POSTGRES__PORT", _PG_PORT)
    monkeypatch.setenv("E4EFS_POSTGRES__USERNAME", _PG_USER)
    monkeypatch.setenv("E4EFS_POSTGRES__PASSWORD", _PG_PASSWORD)
    monkeypatch.setenv("E4EFS_POSTGRES__DATABASE", _SCRATCH_DB)
    settings.reload()
    yield
    monkeypatch.undo()
    settings.reload()


def _admin_dsn(dbname: str = "postgres") -> str:
    return (
        f"host={_PG_HOST} port={_PG_PORT} user={_PG_USER} "
        f"password={_PG_PASSWORD} dbname={dbname}"
    )


@pytest.fixture(scope="module", autouse=True)
def scratch_database():
    """A database of our own, dropped afterwards. Never `fishsense`."""
    import psycopg  # pylint: disable=import-outside-toplevel

    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{_SCRATCH_DB}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{_SCRATCH_DB}"')
    yield
    with psycopg.connect(_admin_dsn(), autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{_SCRATCH_DB}" WITH (FORCE)')


@pytest.fixture
def empty_schema():
    """Reset to a genuinely empty database between tests.

    Dropping the schema rather than the database keeps the connection string
    (and therefore dynaconf's cached settings) stable.
    """
    import psycopg  # pylint: disable=import-outside-toplevel

    with psycopg.connect(_admin_dsn(_SCRATCH_DB), autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    yield


def _run_lifespan_sequence() -> None:
    """Exactly what `fishsense_api.server.lifespan` does, in order."""
    from fishsense_api.database import (  # pylint: disable=import-outside-toplevel
        run_alembic_upgrade,
        setup_database,
    )

    async def _startup():
        database = setup_database()
        async with database.engine.begin() as conn:
            await database.init_database(conn)
        await asyncio.to_thread(run_alembic_upgrade)
        await database.engine.dispose()

    asyncio.run(_startup())


def _query(sql: str):
    import psycopg  # pylint: disable=import-outside-toplevel

    with psycopg.connect(_admin_dsn(_SCRATCH_DB)) as conn:
        return conn.execute(sql).fetchall()


def _views() -> set[str]:
    return {
        row[0]
        for row in _query(
            "SELECT table_name FROM information_schema.views "
            "WHERE table_schema='public'"
        )
    }


def _tables() -> set[str]:
    return {
        row[0]
        for row in _query(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
    }


# ── startup ───────────────────────────────────────────────────────────


@pytest.mark.usefixtures("empty_schema")
def test_startup_on_a_fresh_database_creates_the_tables():
    _run_lifespan_sequence()

    tables = _tables()
    assert "dive" in tables
    assert "image" in tables
    assert "alembic_version" in tables


@pytest.mark.usefixtures("empty_schema")
def test_startup_on_a_fresh_database_creates_every_view():
    """The gap this module was written to find.

    On a fresh DB `run_alembic_upgrade` *stamps* head rather than upgrading,
    because the historical migrations aren't idempotent against `create_all`.
    But the views live only in those migrations and are not part of
    `SQLModel.metadata`, so stamping leaves a database with every table and no
    views — and because head is stamped, it never self-heals: the next restart
    sees `alembic_version`, runs `upgrade head`, and finds nothing to do.

    Prod is unaffected (its DB predates this and has the views), but any fresh
    environment — the local stack, integration CI, a disaster-recovery restore
    into an empty database — silently loses every Superset dashboard.
    """
    from fishsense_api.views import (  # pylint: disable=import-outside-toplevel
        DIVE_PIPELINE_STATUS_VIEW_NAME,
        FISH_LENGTH_ESTIMATE_VIEW_NAME,
        FISH_MODEL_ACCURACY_VIEW_NAME,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME,
    )

    _run_lifespan_sequence()

    assert _views() >= {
        DIVE_PIPELINE_STATUS_VIEW_NAME,
        FISH_MODEL_ACCURACY_VIEW_NAME,
        FISH_LENGTH_ESTIMATE_VIEW_NAME,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME,
    }


@pytest.mark.usefixtures("empty_schema")
def test_every_view_is_actually_queryable_on_postgres():
    """Existing is not the same as valid. These views use Postgres-only
    constructs that SQLite either parses differently or not at all, so a
    malformed one can pass the entire unit suite."""
    from fishsense_api.views import (  # pylint: disable=import-outside-toplevel
        DIVE_PIPELINE_STATUS_VIEW_NAME,
        FISH_LENGTH_ESTIMATE_VIEW_NAME,
        FISH_MODEL_ACCURACY_VIEW_NAME,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME,
    )

    _run_lifespan_sequence()

    for view in (
        DIVE_PIPELINE_STATUS_VIEW_NAME,
        FISH_MODEL_ACCURACY_VIEW_NAME,
        FISH_LENGTH_ESTIMATE_VIEW_NAME,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_NAME,
    ):
        assert _query(f"SELECT * FROM {view} LIMIT 1") == []


@pytest.mark.usefixtures("empty_schema")
def test_startup_is_idempotent():
    """`create_all` runs before `run_alembic_upgrade` on *every* restart, not
    just the first. A restart must be a no-op, not a crash — this is the shape
    of the laserprediction duplicate-table outage."""
    _run_lifespan_sequence()
    _run_lifespan_sequence()
    _run_lifespan_sequence()

    assert "dive" in _tables()
    assert len(_views()) >= 4


@pytest.mark.usefixtures("empty_schema")
def test_a_second_startup_takes_the_upgrade_path_not_the_stamp_path():
    """First run stamps (fresh DB); subsequent runs must find
    `alembic_version` and upgrade. Pins that the branch is chosen on the
    table's presence, so a restart never re-stamps over pending migrations."""
    _run_lifespan_sequence()
    first = _query("SELECT version_num FROM alembic_version")

    _run_lifespan_sequence()

    assert _query("SELECT version_num FROM alembic_version") == first


# ── the views mean what the unit tests claim ──────────────────────────


@pytest.mark.usefixtures("empty_schema")
def test_dive_pipeline_status_reports_a_seeded_dive_on_postgres():
    """The predicates are asserted extensively on SQLite. This checks the same
    SQL against the engine it runs on, with one row through it end to end."""
    _run_lifespan_sequence()

    import psycopg  # pylint: disable=import-outside-toplevel

    with psycopg.connect(_admin_dsn(_SCRATCH_DB), autocommit=True) as conn:
        conn.execute(
            "INSERT INTO dive (id, path, dive_datetime, priority) "
            "VALUES (1, 'd/1', '2024-08-21 08:56:51+00', 'HIGH')"
        )

    rows = _query(
        "SELECT dive_id, laser_preprocessed, measured "
        "FROM dive_pipeline_status WHERE dive_id = 1"
    )

    assert len(rows) == 1
    # Vacuous truth reads as False throughout — a dive with no labels at all is
    # not "complete". Same rule the SQLite tests pin.
    assert rows[0][2] is False


# ── regressions found by review of the first cut ──────────────────────


@pytest.mark.usefixtures("empty_schema")
def test_recreating_the_views_when_they_already_exist_does_not_fail():
    """`fish_model_species_mislabel_suspects` selects from
    `fish_model_measurement_accuracy`, so dropping the latter while the former
    exists raises `DependentObjectsStillExistError`.

    The original per-view drop-then-create loop hit exactly that on any second
    run, which meant the "safe to re-run" claim was false and a startup on a
    database with views but no `alembic_version` would crash-loop the API.
    Dropping everything in reverse order first fixes it — and this is the test
    that would have caught it, because SQLite has no such dependency tracking.
    """
    from fishsense_api.database import (  # pylint: disable=import-outside-toplevel
        _create_all_views,
    )

    _run_lifespan_sequence()
    assert len(_views()) >= 4

    asyncio.run(_create_all_views())      # must not raise
    asyncio.run(_create_all_views())

    assert len(_views()) >= 4


@pytest.mark.usefixtures("empty_schema")
def test_an_already_stamped_database_with_no_views_is_repaired():
    """The state the previous implementation left behind.

    `alembic_version` is present, so startup takes the upgrade branch and
    alembic finds nothing to do. Without an explicit repair those databases
    stay viewless permanently.
    """
    import psycopg  # pylint: disable=import-outside-toplevel

    _run_lifespan_sequence()

    with psycopg.connect(_admin_dsn(_SCRATCH_DB), autocommit=True) as conn:
        for view in ("fish_model_species_mislabel_suspects", "dive_pipeline_status"):
            conn.execute(f"DROP VIEW {view}")
    assert "dive_pipeline_status" not in _views()

    _run_lifespan_sequence()

    assert "dive_pipeline_status" in _views()
    assert "fish_model_species_mislabel_suspects" in _views()


@pytest.mark.usefixtures("empty_schema")
def test_a_healthy_database_is_not_rebuilt_on_every_restart():
    """The repair must be conditional. Dropping and recreating every view on
    each startup would break whatever Superset had mid-query, for no reason."""
    from fishsense_api import database as db_module  # pylint: disable=import-outside-toplevel

    _run_lifespan_sequence()

    calls = []
    original = db_module._create_all_views  # pylint: disable=protected-access

    async def _counting():
        calls.append(1)
        await original()

    db_module._create_all_views = _counting  # pylint: disable=protected-access
    try:
        _run_lifespan_sequence()
    finally:
        db_module._create_all_views = original  # pylint: disable=protected-access

    assert not calls


# ── cohort selectors, on the database they actually run against ───────
#
# The unit suite exercises these on in-memory SQLite, which cannot see two
# whole classes of bug that prod hit within a day of each other:
#
#   * SQLite answers a multi-row scalar subquery with its first row; Postgres
#     raises CardinalityViolationError. An uncorrelated
#     `_resolved_laser_extrinsics_id()` therefore passed every unit test and
#     500'd both selectors in prod on every hourly poll.
#   * Every unit fixture seeds one row where prod has many — one
#     `laserextrinsics` per dive, one valid laser label per image. Prod has
#     461 images carrying two valid labels, which wedged the laser-depth
#     cohort on dive 279 forever.
#
# So the seed below is deliberately *multi-valued*: two calibrated dives and
# an image with two valid laser labels. Both bugs are visible here and
# invisible on a single-row SQLite fixture.


def _cohort_selectors() -> list:
    """Every cohort selector, discovered rather than listed.

    A registry property in the spirit of
    `test_canonical_only_pipeline_work.py`: a selector added later is covered
    by this the day it lands, instead of the day someone remembers to add it.
    """
    import inspect  # pylint: disable=import-outside-toplevel

    from fishsense_api.controllers import (  # pylint: disable=import-outside-toplevel
        dive_cohort_controller,
    )

    return sorted(
        (name, fn)
        for name, fn in inspect.getmembers(dive_cohort_controller, inspect.iscoroutinefunction)
        if name.startswith("select_next_for_") or name.startswith("select_dives_")
    )


async def _seed_multi_valued(session, *, derived_rows: bool = True) -> None:
    """A dive population with the multiplicity prod actually has.

    `derived_rows` controls whether a `LaserDepth` and a `Measurement` already
    exist. They must, for the selector-executes test: without them the
    selectors' `NOT EXISTS` short-circuits and the scalar subquery inside is
    never evaluated. The drain test needs them absent so it can watch a dive
    leave the cohort.
    """
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.camera import Camera  # pylint: disable=import-outside-toplevel

    when = datetime(2025, 1, 1, tzinfo=timezone.utc)
    # Postgres enforces the foreign keys the SQLite fixtures quietly ignore, so
    # the camera has to exist before anything references it.
    session.add(Camera(id=1, serial_number="itest-0001", name="itest-camera"))
    await session.flush()
    for dive_id in (1, 2):
        session.add(
            Dive(id=dive_id, path=f"/dev/null/{dive_id}", dive_datetime=when,
                 priority=Priority.HIGH)
        )
    await session.flush()
    # TWO extrinsics rows: one per dive. This is what makes an uncorrelated
    # scalar subquery return more than one row.
    session.add_all([
        LaserExtrinsics(id=51, laser_position=[0.1, 0.0, 0.0], laser_axis=[0.0, 0.0, 1.0],
                        dive_id=1, camera_id=1),
        LaserExtrinsics(id=52, laser_position=[0.1, 0.0, 0.0], laser_axis=[0.0, 0.0, 1.0],
                        dive_id=2, camera_id=1),
    ])
    session.add(Image(id=11, path="/dev/null/img-11", taken_datetime=when,
                      checksum=f"{11:032d}", is_canonical=True, dive_id=1))
    await session.flush()
    # TWO valid laser labels on one image — duplicates of the same dot, the
    # shape 461 prod images have.
    session.add_all([
        LaserLabel(id=101, x=100.0, y=200.0, completed=True, superseded=False, image_id=11),
        LaserLabel(id=102, x=100.0, y=200.0, completed=True, superseded=False, image_id=11),
    ])
    await session.flush()
    # A depth and a measurement must EXIST, or the selectors' `NOT EXISTS`
    # short-circuits and the scalar subquery inside it is never evaluated —
    # which is exactly how an uncorrelated `coalesce(...)` slips through: the
    # query runs, returns a row, and Postgres never gets the chance to raise
    # CardinalityViolationError.
    if not derived_rows:
        return

    from fishsense_api.models.fish import Fish  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_depth import LaserDepth  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    session.add(Fish(id=700, species_id=None))
    await session.flush()
    session.add_all([
        LaserDepth(id=1, depth_m=1.5, range_m=1.51, residual_m=0.0004,
                   image_id=11, laser_label_id=101, laser_extrinsics_id=51),
        Measurement(id=1, length_m=0.3, image_id=11, fish_id=700,
                    laser_extrinsics_id=51),
    ])
    await session.flush()


def _with_session(coro_factory):
    """Run `coro_factory(session)` against the scratch database."""
    from fishsense_api.database import setup_database  # pylint: disable=import-outside-toplevel
    from sqlalchemy.ext.asyncio import async_sessionmaker  # pylint: disable=import-outside-toplevel
    from sqlmodel.ext.asyncio.session import AsyncSession  # pylint: disable=import-outside-toplevel

    async def _run():
        database = setup_database()
        factory = async_sessionmaker(database.engine, class_=AsyncSession,
                                     expire_on_commit=False)
        try:
            async with factory() as session:
                result = await coro_factory(session)
                # Commit, or nothing survives the session. Seeding in one
                # session and querying in another silently saw an empty
                # database, so the selectors' `NOT EXISTS` short-circuited and
                # never evaluated the scalar subquery this test exists to
                # exercise — it passed with the bug reintroduced.
                await session.commit()
                return result
        finally:
            await database.engine.dispose()

    return asyncio.run(_run())


@pytest.mark.usefixtures("empty_schema")
def test_every_cohort_selector_runs_on_postgres():
    """Each selector must execute against real Postgres without raising.

    This is the test that would have caught the CardinalityViolation outage
    before it shipped: on SQLite the same query returns a row and passes.
    """
    _run_lifespan_sequence()
    selectors = _cohort_selectors()
    assert selectors, "no cohort selectors discovered - has the module moved?"

    _with_session(_seed_multi_valued)

    def _exercise(name, fn):
        # A session each: the first Postgres error aborts its transaction, and
        # every later statement on that connection comes back
        # InFailedSqlTransaction — which would report every selector as broken
        # and hide which one actually is.
        async def _run(session):
            try:
                await fn(session=session)
                return None
            except Exception as exc:  # noqa: BLE001  pylint: disable=broad-except
                return f"{name}: {type(exc).__name__}: {exc}"

        return _with_session(_run)

    failures = [f for f in (_exercise(name, fn) for name, fn in selectors) if f]

    assert not failures, "cohort selectors raised on Postgres:\n  " + "\n  ".join(failures)


@pytest.mark.usefixtures("empty_schema")
def test_laser_depth_cohort_drains_with_duplicate_labels_on_postgres():
    """The dive-279 wedge, end to end on Postgres.

    One depth row covers the image even though a second valid label exists —
    otherwise the dive is offered forever and blocks every higher-id dive.
    """
    _run_lifespan_sequence()

    async def _exercise(session):
        from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
            select_next_for_laser_depth,
        )
        from fishsense_api.controllers.laser_depth_controller import (  # pylint: disable=import-outside-toplevel
            put_laser_depth,
        )
        from fishsense_api.models.laser_depth import LaserDepth  # pylint: disable=import-outside-toplevel

        await _seed_multi_valued(session, derived_rows=False)
        before = await select_next_for_laser_depth(session=session)
        await put_laser_depth(
            11,
            LaserDepth(depth_m=1.5, range_m=1.51, residual_m=0.0004,
                       laser_label_id=101, laser_extrinsics_id=51),
            session=session,
        )
        after = await select_next_for_laser_depth(session=session)
        return before, after

    before, after = _with_session(_exercise)
    assert before == 1, "dive with an undepthed laser image should be offered"
    assert after is None, (
        "dive still offered after its only eligible image got a depth — the "
        "cohort is keyed on the label rather than the image and will never drain"
    )

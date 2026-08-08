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

# Must be set BEFORE anything imports `fishsense_api.config`: dynaconf reads
# the environment on first attribute access and caches it, and
# `run_alembic_upgrade` resolves its URL through `pg_connection_string()`.
_PG_HOST = os.environ.get("FISHSENSE_POSTGRES_HOST", "postgres")
_PG_PORT = os.environ.get("FISHSENSE_POSTGRES_PORT", "5432")
_PG_USER = os.environ.get("FISHSENSE_POSTGRES_USER", "postgres")
_PG_PASSWORD = os.environ.get("FISHSENSE_POSTGRES_PASSWORD", "fishsense_local")
_SCRATCH_DB = os.environ.get("FISHSENSE_POSTGRES_ITEST_DB", "fishsense_api_itest")

os.environ["E4EFS_POSTGRES__HOST"] = _PG_HOST
os.environ["E4EFS_POSTGRES__PORT"] = _PG_PORT
os.environ["E4EFS_POSTGRES__USERNAME"] = _PG_USER
os.environ["E4EFS_POSTGRES__PASSWORD"] = _PG_PASSWORD
os.environ["E4EFS_POSTGRES__DATABASE"] = _SCRATCH_DB

pytestmark = pytest.mark.integration


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

"""The `Dive.notes` + `Priority.NONE` migration.

Two halves with very different failure modes:

* `notes` is an ordinary `add_column`, portable everywhere.
* `NONE` is a new label on the Postgres native enum type `priority`, which
  needs `ALTER TYPE ... ADD VALUE`. That statement does not exist on SQLite
  (where SQLModel renders the enum as a VARCHAR + CHECK), so it has to be
  dialect-guarded or every migration test in this suite dies on it.

The `IF NOT EXISTS` is load-bearing rather than defensive. `lifespan` runs
`SQLModel.metadata.create_all` *before* `run_alembic_upgrade`, so on a fresh
database the enum type is created from the model — already carrying `NONE` —
and the migration then runs against it. Without `IF NOT EXISTS` that path
raises `DuplicateObject` and the API never finishes starting.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa


@pytest.fixture
def migration():
    from fishsense_api.alembic.versions import (
        b3d5e91a7c42_add_dive_notes_and_none_priority as mod,
    )

    return mod


@pytest.fixture
def engine():
    """A `dive` table shaped like the pre-migration schema."""
    eng = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    table = sa.Table(
        "dive",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("path", sa.String(255)),
        sa.Column("dive_datetime", sa.DateTime(timezone=True)),
        sa.Column("priority", sa.String()),
    )
    metadata.create_all(eng)
    return eng, table


def _run_upgrade(migration, conn):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()


def _columns(conn) -> set[str]:
    return {c["name"] for c in sa.inspect(conn).get_columns("dive")}


# --- the portable half -----------------------------------------------------


def test_adds_the_notes_column(migration, engine):
    eng, _ = engine
    with eng.begin() as conn:
        _run_upgrade(migration, conn)
        assert "notes" in _columns(conn)


def test_notes_is_nullable_so_existing_rows_survive(migration, engine):
    """Every dive in prod predates this column; a NOT NULL would fail the
    migration outright on a non-empty table."""
    eng, table = engine
    with eng.begin() as conn:
        conn.execute(
            table.insert().values(
                id=1,
                path="d/1",
                dive_datetime=datetime(2024, 8, 21, tzinfo=timezone.utc),
                priority="HIGH",
            )
        )
        _run_upgrade(migration, conn)

        assert conn.execute(sa.text("SELECT notes FROM dive")).scalar() is None


def test_upgrade_does_not_raise_on_sqlite(migration, engine):
    """The dialect guard, stated as a test.

    Without it the Postgres-only `ALTER TYPE` reaches SQLite and every
    migration test in the suite fails on an unrelated change.
    """
    eng, _ = engine
    with eng.begin() as conn:
        _run_upgrade(migration, conn)   # must not raise


# --- the Postgres-only half ------------------------------------------------


def test_enum_sql_is_emitted_for_postgres(migration):
    sql = migration.add_enum_value_sql("postgresql")

    assert sql is not None
    normalized = " ".join(sql.split()).upper()
    assert "ALTER TYPE" in normalized
    assert "PRIORITY" in normalized
    assert "'NONE'" in normalized


def test_enum_sql_is_idempotent(migration):
    """`create_all` may already have built the type with NONE in it."""
    sql = migration.add_enum_value_sql("postgresql")

    assert "IF NOT EXISTS" in " ".join(sql.split()).upper()


@pytest.mark.parametrize("dialect", ["sqlite", "mysql"])
def test_no_enum_sql_off_postgres(migration, dialect):
    assert migration.add_enum_value_sql(dialect) is None

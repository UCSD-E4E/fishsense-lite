"""Tests for the `calibrationtarget` table + `dive.calibration_target_id`.

Two properties, and both have already failed in this repo for other
migrations:

  * **Idempotent against `create_all`.** `lifespan` runs
    `SQLModel.metadata.create_all` *before* `run_alembic_upgrade`, so on a
    fresh database the table and the column already exist by the time this
    migration runs. A bare `create_table` / `add_column` would raise and stop
    the API from starting.
  * **`square_size_m` is NOT NULL in the DDL the migration writes**, not just
    in the ORM. The migration is what an existing production database
    actually gets.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MODULE = (
    "fishsense_api.alembic.versions."
    "d92a1f4c78b3_add_calibration_target_and_dive_link"
)


def _upgrade(connection) -> None:
    import importlib

    module = importlib.import_module(MODULE)
    context = MigrationContext.configure(connection)
    with Operations.context(context):
        module.upgrade()


def test_upgrade_creates_the_table_and_the_link():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dive (id INTEGER PRIMARY KEY, path VARCHAR NOT NULL)"
            )
        )
        _upgrade(connection)

        inspector = sa.inspect(connection)
        assert "calibrationtarget" in inspector.get_table_names()
        columns = {c["name"]: c for c in inspector.get_columns("calibrationtarget")}
        assert set(columns) >= {
            "id",
            "name",
            "rows",
            "cols",
            "square_size_m",
            "notes",
            "created_at",
        }
        assert "calibration_target_id" in {
            c["name"] for c in inspector.get_columns("dive")
        }


def test_the_scale_column_is_not_null():
    """The number that sets every length's scale cannot be left unsaid."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dive (id INTEGER PRIMARY KEY, path VARCHAR NOT NULL)"
            )
        )
        _upgrade(connection)

        columns = {
            c["name"]: c for c in sa.inspect(connection).get_columns("calibrationtarget")
        }
        for required in ("name", "rows", "cols", "square_size_m"):
            assert columns[required]["nullable"] is False


def test_upgrade_is_idempotent_against_create_all():
    """The fresh-database path: both artifacts already exist.

    `create_all` runs first in `lifespan`, so this is the ordinary startup
    sequence on a new deployment, not an edge case.
    """
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dive (id INTEGER PRIMARY KEY, path VARCHAR NOT NULL, "
                "calibration_target_id INTEGER)"
            )
        )
        connection.execute(
            sa.text(
                "CREATE TABLE calibrationtarget ("
                "id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, "
                "rows INTEGER NOT NULL, cols INTEGER NOT NULL, "
                "square_size_m FLOAT NOT NULL, notes VARCHAR, created_at DATETIME)"
            )
        )

        _upgrade(connection)  # must not raise

        assert "calibrationtarget" in sa.inspect(connection).get_table_names()


def test_downgrade_drops_both():
    import importlib

    module = importlib.import_module(MODULE)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE dive (id INTEGER PRIMARY KEY, path VARCHAR NOT NULL)"
            )
        )
        _upgrade(connection)

        context = MigrationContext.configure(connection)
        with Operations.context(context):
            module.downgrade()

        inspector = sa.inspect(connection)
        assert "calibrationtarget" not in inspector.get_table_names()
        assert "calibration_target_id" not in {
            c["name"] for c in inspector.get_columns("dive")
        }

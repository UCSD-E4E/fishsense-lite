"""`f2b8c04e71a3` — the provisional flag, and the notes the seed can miss.

Both behaviours are exercised against a real SQLite database rather than mocks,
because the bug being fixed is an *ordering* one between migrations that each
look correct in isolation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa

_VERSIONS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fishsense_api"
    / "alembic"
    / "versions"
)

NAME = "Weasly Fish"


def _load(filename: str, mod: str):
    spec = importlib.util.spec_from_file_location(mod, _VERSIONS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seed_migration():
    return _load("d1a7b3e95c02_seed_weasly_fish_reference.py", "seed_mig")


@pytest.fixture
def provisional_migration():
    return _load(
        "f2b8c04e71a3_fish_model_reference_is_provisional.py", "prov_mig"
    )


@pytest.fixture
def conn():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as c:
        # Shaped as the EARLIER migrations leave it: no is_provisional column.
        c.execute(
            sa.text(
                "CREATE TABLE fishmodelreference ("
                " id INTEGER PRIMARY KEY,"
                " name TEXT NOT NULL UNIQUE,"
                " known_length_m REAL NOT NULL,"
                " notes TEXT)"
            )
        )
        yield c


def _run(mod, conn, fn="upgrade"):
    fake_op = MagicMock()
    fake_op.get_bind.return_value = conn

    def _add_column(table, col):
        conn.execute(
            sa.text(
                f"ALTER TABLE {table} ADD COLUMN {col.name} BOOLEAN "
                "NOT NULL DEFAULT 0"
            )
        )

    def _drop_column(table, name):
        conn.execute(sa.text(f"ALTER TABLE {table} DROP COLUMN {name}"))

    fake_op.add_column.side_effect = _add_column
    fake_op.drop_column.side_effect = _drop_column
    fake_op.execute.side_effect = lambda sql: conn.execute(sa.text(str(sql)))
    original = mod.op
    mod.op = fake_op
    try:
        getattr(mod, fn)()
    finally:
        mod.op = original


def _row(conn):
    rows = [
        dict(r._mapping)  # pylint: disable=protected-access
        for r in conn.execute(
            sa.text(
                "SELECT known_length_m, notes, is_provisional "
                "FROM fishmodelreference WHERE name = :n"
            ),
            {"n": NAME},
        )
    ]
    return rows[0] if rows else None


def test_backfills_notes_an_earlier_migration_inserted_without(
    provisional_migration, conn
):
    """The ordering bug this migration exists to repair.

    `e2c9a4f70b31` and `b4c81f60d7e2` iterate the LIVE `KNOWN_FISH_MODELS`, so
    once Weasly Fish was added to that constant they began inserting it
    themselves — with `notes = NULL`, since their INSERT binds only
    (name, known_length_m). `d1a7b3e95c02` then finds the row present and
    short-circuits, so on any database below those revisions the caliper
    provenance would be missing permanently.
    """
    conn.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m) "
            "VALUES (:n, 0.310)"
        ),
        {"n": NAME},
    )

    _run(provisional_migration, conn)

    row = _row(conn)
    assert "58.69" in row["notes"]
    assert "29.56" in row["notes"]


def test_marks_the_provisional_row_and_leaves_measured_ones_alone(
    provisional_migration, conn
):
    conn.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m) "
            "VALUES (:n, 0.310), ('Grouper', 0.360)"
        ),
        {"n": NAME},
    )

    _run(provisional_migration, conn)

    assert _row(conn)["is_provisional"] == 1
    grouper = conn.execute(
        sa.text(
            "SELECT is_provisional FROM fishmodelreference WHERE name='Grouper'"
        )
    ).scalar_one()
    assert grouper == 0


def test_never_overwrites_notes_an_operator_wrote(provisional_migration, conn):
    conn.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m, notes) "
            "VALUES (:n, 0.287, 'calipered 2026-09-01')"
        ),
        {"n": NAME},
    )

    _run(provisional_migration, conn)

    assert _row(conn)["notes"] == "calipered 2026-09-01"


def test_running_twice_is_safe(provisional_migration, conn):
    """`run_alembic_upgrade` runs on every start; the column add must not
    explode on the second pass."""
    conn.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m) "
            "VALUES (:n, 0.310)"
        ),
        {"n": NAME},
    )

    _run(provisional_migration, conn)
    _run(provisional_migration, conn)

    assert _row(conn)["is_provisional"] == 1


def test_the_seed_then_provisional_chain_ends_with_both(
    seed_migration, provisional_migration, conn
):
    """End to end, in the order a real database runs them."""
    _run(seed_migration, conn)
    _run(provisional_migration, conn)

    row = _row(conn)
    assert row["known_length_m"] == pytest.approx(0.310)
    assert row["is_provisional"] == 1
    assert "58.69" in row["notes"]

"""The `Weasly Fish` seed migration, run against a real SQLite database.

Mocks would prove the code calls `execute`; this proves the SQL it emits does
what the migration claims. The behaviour that matters is **seed-only-if-absent**
— an operator may have corrected a length by hand, and a migration that stamps
over that silently replaces a measured value with an estimate.
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


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location(
        "weasly_seed_migration", _VERSIONS / "d1a7b3e95c02_seed_weasly_fish_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conn():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as c:
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


def _run(migration, monkeypatch, conn, fn="upgrade"):
    fake_op = MagicMock()
    fake_op.get_bind.return_value = conn
    monkeypatch.setattr(migration, "op", fake_op)
    getattr(migration, fn)()


def _rows(conn):
    return [
        dict(r._mapping)  # pylint: disable=protected-access
        for r in conn.execute(
            sa.text(
                "SELECT name, known_length_m, notes FROM fishmodelreference "
                "WHERE name = :n"
            ),
            {"n": NAME},
        )
    ]


def test_seeds_the_row_with_its_length_and_notes(migration, monkeypatch, conn):
    _run(migration, monkeypatch, conn)

    rows = _rows(conn)
    assert len(rows) == 1
    assert rows[0]["known_length_m"] == pytest.approx(0.310)
    assert "58.69" in rows[0]["notes"]
    assert "29.56" in rows[0]["notes"]


def test_running_twice_leaves_one_row(migration, monkeypatch, conn):
    """`run_alembic_upgrade` is idempotent by design and runs on every start;
    a UNIQUE violation here would crash the API at boot."""
    _run(migration, monkeypatch, conn)
    _run(migration, monkeypatch, conn)

    assert len(_rows(conn)) == 1


def test_does_not_overwrite_a_hand_corrected_length(migration, monkeypatch, conn):
    """The length being seeded is an ESTIMATE. If someone has since calipered
    the model and corrected the row, this migration must leave it alone —
    otherwise a deploy silently replaces a measurement with a guess."""
    conn.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m, notes) "
            "VALUES (:n, 0.287, 'calipered 2026-09-01')"
        ),
        {"n": NAME},
    )

    _run(migration, monkeypatch, conn)

    rows = _rows(conn)
    assert len(rows) == 1
    assert rows[0]["known_length_m"] == pytest.approx(0.287)
    assert rows[0]["notes"] == "calipered 2026-09-01"


def test_downgrade_leaves_the_row_alone(migration, monkeypatch, conn):
    """A downgrade must not discard reference data.

    The row is inert without a measurement to grade, and `upgrade()` explicitly
    refuses to stamp over a hand-corrected length — deleting it here would throw
    away exactly the work that guard protects. Sibling `b4c81f60d7e2` seeds the
    same table and takes the same position, in as many words.

    This test previously asserted the opposite and pinned the unsafe behaviour.
    """
    _run(migration, monkeypatch, conn)
    conn.execute(
        sa.text(
            "UPDATE fishmodelreference SET known_length_m = 0.287, "
            "notes = 'calipered later' WHERE name = :n"
        ),
        {"n": NAME},
    )

    _run(migration, monkeypatch, conn, fn="downgrade")

    rows = _rows(conn)
    assert len(rows) == 1
    assert rows[0]["known_length_m"] == pytest.approx(0.287)
    assert rows[0]["notes"] == "calipered later"


def test_upgrade_is_a_no_op_if_the_model_is_ever_renamed(
    migration, monkeypatch, conn
):
    """This migration replays on every fresh database forever. A bare
    `next(...)` over `KNOWN_FISH_MODELS` would raise StopIteration inside the
    FastAPI lifespan after a rename, failing API boot — so it must degrade to
    doing nothing instead."""
    monkeypatch.setattr(
        migration,
        "KNOWN_FISH_MODELS",
        [{"name": "Something Else", "known_length_m": 0.1}],
    )

    _run(migration, monkeypatch, conn)

    assert _rows(conn) == []

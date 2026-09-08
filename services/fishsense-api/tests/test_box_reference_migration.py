"""The `Box` seed migration, run against a real SQLite database.

Same shape and same reason as `test_weasly_fish_seed_migration.py`: mocks would
prove the migration calls `execute`, this proves the SQL it emits does what the
migration claims. The behaviour that matters is **seed-only-if-absent** — an
operator who has re-measured the box by hand must not have that stamped back to
the committed 0.15 m on the next upgrade.

The migration also drops and recreates `dive_pipeline_status`, because
`rigid_target_sql` widened to include the box. That half is asserted here too:
a seed without the predicate change leaves box frames ungraded, and a predicate
change without the seed offers the stage-14 cohort frames whose measurements
the accuracy view inner-joins away.
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

NAME = "Box"


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location(
        "box_seed_migration",
        _VERSIONS / "a1c6d0f483b7_seed_box_reference_and_widen_measurable.py",
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
                " notes TEXT,"
                " is_provisional BOOLEAN NOT NULL DEFAULT 0)"
            )
        )
        yield c


def _run(migration, monkeypatch, conn, fn="upgrade"):
    fake_op = MagicMock()
    fake_op.get_bind.return_value = conn
    monkeypatch.setattr(migration, "op", fake_op)
    getattr(migration, fn)()
    return fake_op


def _rows(conn, name=NAME):
    return [
        dict(r._mapping)  # pylint: disable=protected-access
        for r in conn.execute(
            sa.text(
                "SELECT name, known_length_m, notes, is_provisional "
                "FROM fishmodelreference WHERE name = :n"
            ),
            {"n": name},
        )
    ]


def test_seeds_the_box_at_its_known_length(migration, monkeypatch, conn):
    _run(migration, monkeypatch, conn)

    rows = _rows(conn)
    assert len(rows) == 1
    assert rows[0]["known_length_m"] == pytest.approx(0.15)


def test_the_box_is_not_provisional(migration, monkeypatch, conn):
    """0.15 m is a measurement the operator supplied, not an estimate — and
    `is_provisional` excludes a row from the mislabel view's best-fit search,
    so flagging it would quietly change what that view can attribute."""
    _run(migration, monkeypatch, conn)

    assert not _rows(conn)[0]["is_provisional"]


def test_the_row_records_where_the_length_came_from(migration, monkeypatch, conn):
    """A reference with no provenance is one nobody can re-check, and the
    ruler's 14-vs-13.5 in history is what that costs."""
    _run(migration, monkeypatch, conn)

    assert _rows(conn)[0]["notes"]


def test_seeding_does_not_overwrite_a_hand_corrected_length(
    migration, monkeypatch, conn
):
    conn.execute(
        sa.text(
            "INSERT INTO fishmodelreference (name, known_length_m, notes) "
            "VALUES (:n, 0.1487, 'calipered')"
        ),
        {"n": NAME},
    )

    _run(migration, monkeypatch, conn)

    rows = _rows(conn)
    assert len(rows) == 1
    assert rows[0]["known_length_m"] == pytest.approx(0.1487)
    assert rows[0]["notes"] == "calipered"


def test_it_is_idempotent(migration, monkeypatch, conn):
    _run(migration, monkeypatch, conn)
    _run(migration, monkeypatch, conn)

    assert len(_rows(conn)) == 1


def test_it_recreates_both_affected_views(migration, monkeypatch, conn):
    """Two views move, for two different reasons.

    `dive_pipeline_status` because the measurable predicate widened — without
    the rebuild the dashboard keeps reporting box dives as having nothing to
    measure. `fish_model_species_mislabel_suspects` because the seeded Box row
    lands in its CROSS JOIN, where at 0.150 m it would flag correct-but-angled
    frames of the ~0.195 m models; the recreated SQL excludes calibration
    targets from that search.

    The accuracy view is deliberately absent: it joins on `Fish.name` and needs
    no change for Box to grade.
    """
    from fishsense_api.views import (
        DIVE_PIPELINE_STATUS_VIEW_SQL,
        DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
        DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    )

    fake_op = _run(migration, monkeypatch, conn)

    executed = [call.args[0] for call in fake_op.execute.call_args_list]
    assert executed == [
        DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
        DIVE_PIPELINE_STATUS_VIEW_SQL,
        DROP_FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
        FISH_MODEL_MISLABEL_SUSPECTS_VIEW_SQL,
    ]


def test_downgrade_rebuilds_the_view_and_keeps_the_row(migration, monkeypatch, conn):
    """The seeded row is left in place on the way down: it is inert without the
    predicate, and dropping it would discard a hand-corrected span."""
    _run(migration, monkeypatch, conn)

    _run(migration, monkeypatch, conn, fn="downgrade")

    assert len(_rows(conn)) == 1

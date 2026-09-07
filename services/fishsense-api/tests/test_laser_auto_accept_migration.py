"""The laser-prediction auto-accept verdict migration.

`auto_accept` gets `server_default=false` + NOT NULL for the same reason
`rejected_out_of_region` and `needs_reprocess` did: it is read as a predicate,
and a NULL backfill makes `WHERE NOT auto_accept` silently skip every
pre-existing row under three-valued logic. Here that failure has a direction
that matters — the laser populate step decides whether a human ever sees the
frame — so the backfill must land on False, meaning "the gate has not judged
this row, send it to a person".

The other three stay nullable on purpose. NULL means the gate never ran, which
is a state every row predating this migration is genuinely in, and it must stay
distinguishable from a gate that ran and said no.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa


@pytest.fixture
def migration():
    from fishsense_api.alembic.versions import (
        c4f8a2e60b17_add_laser_prediction_auto_accept as mod,
    )

    return mod


@pytest.fixture
def engine():
    eng = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    table = sa.Table(
        "laserprediction",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer),
        sa.Column("confidence", sa.Float),
    )
    metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(table.insert().values(image_id=1, confidence=0.9))
    return eng


def _run(migration, conn, direction="upgrade"):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with Operations.context(MigrationContext.configure(conn)):
        getattr(migration, direction)()


def _columns(conn):
    return {c["name"]: c for c in sa.inspect(conn).get_columns("laserprediction")}


_NEW = {"auto_accept", "gate_verdict", "line_offset_px", "line_position_z"}


def test_adds_all_four_columns(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        assert _NEW <= set(_columns(conn))


def test_existing_rows_are_not_auto_acceptable(migration, engine):
    """The whole safety argument rests on this default. A row the gate has
    never seen must route to a human."""
    with engine.begin() as conn:
        _run(migration, conn)
        value = conn.execute(
            sa.text("SELECT auto_accept FROM laserprediction WHERE image_id=1")
        ).scalar_one()
        assert value in (0, False)


def test_auto_accept_is_not_null_but_the_verdict_columns_are(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        cols = _columns(conn)
        assert not cols["auto_accept"]["nullable"]
        assert cols["gate_verdict"]["nullable"]
        assert cols["line_offset_px"]["nullable"]
        assert cols["line_position_z"]["nullable"]


def test_upgrade_is_idempotent_against_create_all(migration, engine):
    """`lifespan` runs `SQLModel.metadata.create_all` BEFORE alembic, so on a
    fresh database the columns already exist when this runs. A bare add_column
    would raise DuplicateColumn and stop the API from starting."""
    with engine.begin() as conn:
        _run(migration, conn)
        _run(migration, conn)
        assert _NEW <= set(_columns(conn))


def test_downgrade_removes_all_four(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        _run(migration, conn, "downgrade")
        assert _NEW.isdisjoint(_columns(conn))

"""The laser-prediction colour / out-of-region migration.

`rejected_out_of_region` gets `server_default=false` + NOT NULL for the same
reason `needs_reprocess` did: it is read as a predicate, and a NULL backfill
makes `WHERE NOT rejected_out_of_region` silently skip every pre-existing row
under three-valued logic.

`color` is deliberately the opposite — nullable. NULL means "the classifier had
no opinion", which is a state it genuinely reaches (no dot to sample, or the
channels too close to call), and collapsing that into "red" would put a
confident wrong colour on every frame it could not read.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa


@pytest.fixture
def migration():
    from fishsense_api.alembic.versions import (
        d81b6c4a5f27_add_laser_prediction_color_and_region as mod,
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


def test_adds_all_three_columns(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        cols = _columns(conn)
        assert {"color", "color_margin", "rejected_out_of_region"} <= set(cols)


def test_existing_rows_backfill_false_not_null(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        value = conn.execute(
            sa.text("SELECT rejected_out_of_region FROM laserprediction WHERE image_id=1")
        ).scalar_one()
        assert value in (0, False)


def test_colour_columns_stay_nullable(migration, engine):
    """NULL is a meaningful value here: 'no opinion', not 'red'."""
    with engine.begin() as conn:
        _run(migration, conn)
        cols = _columns(conn)
        assert cols["color"]["nullable"]
        assert cols["color_margin"]["nullable"]
        assert not cols["rejected_out_of_region"]["nullable"]


def test_downgrade_removes_all_three(migration, engine):
    with engine.begin() as conn:
        _run(migration, conn)
        _run(migration, conn, "downgrade")
        cols = set(_columns(conn))
        assert not ({"color", "color_margin", "rejected_out_of_region"} & cols)

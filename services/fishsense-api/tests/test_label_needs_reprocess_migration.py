"""The `needs_reprocess` migration across the four label tables.

An ordinary `add_column` x4, so the interesting part is not that it runs but
what it leaves behind on the rows that were already there. The column is the
input to a cohort predicate, and a NULL backfill would make
`WHERE NOT needs_reprocess` silently skip every pre-existing label under
three-valued logic — a dive would look clean and never be redrawn. Hence
`server_default=false` + NOT NULL, asserted here against rows written *before*
the upgrade.

Same failure shape as the `laserextrinsics.created_at` NULL bug, where
Postgres' NULLS-FIRST `ORDER BY ... DESC` made a latest-wins read match
nothing at all.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa


@pytest.fixture
def migration():
    from fishsense_api.alembic.versions import (
        c7e4a91f2d38_add_needs_reprocess_to_label_tables as mod,
    )

    return mod


@pytest.fixture
def engine(migration):
    """The four label tables shaped like the pre-migration schema, each
    carrying a row written before the column existed."""
    eng = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    tables = {}
    for name in migration.LABEL_TABLES:
        tables[name] = sa.Table(
            name,
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("image_id", sa.Integer),
            sa.Column("label_studio_project_id", sa.Integer),
            sa.Column("superseded", sa.Boolean()),
        )
    metadata.create_all(eng)
    with eng.begin() as conn:
        for name, table in tables.items():
            conn.execute(table.insert().values(image_id=1, superseded=False))
    return eng, tables


def _run(migration, conn, direction="upgrade"):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with Operations.context(MigrationContext.configure(conn)):
        getattr(migration, direction)()


def test_covers_every_label_table(migration):
    """All four, not the three that happened to come to mind."""
    assert set(migration.LABEL_TABLES) == {
        "laserlabel",
        "specieslabel",
        "headtaillabel",
        "diveslatelabel",
    }


def test_adds_the_column_to_every_table(migration, engine):
    eng, _ = engine
    with eng.begin() as conn:
        _run(migration, conn)
        for name in migration.LABEL_TABLES:
            cols = {c["name"] for c in sa.inspect(conn).get_columns(name)}
            assert "needs_reprocess" in cols, name


def test_backfills_existing_rows_false_not_null(migration, engine):
    """The whole reason for `server_default`."""
    eng, _ = engine
    with eng.begin() as conn:
        _run(migration, conn)
        for name in migration.LABEL_TABLES:
            value = conn.execute(
                sa.text(f"SELECT needs_reprocess FROM {name} WHERE image_id = 1")  # nosec
            ).scalar_one()
            assert value in (0, False), f"{name} backfilled {value!r}, expected false"


def test_column_is_not_nullable(migration, engine):
    eng, _ = engine
    with eng.begin() as conn:
        _run(migration, conn)
        for name in migration.LABEL_TABLES:
            col = next(
                c
                for c in sa.inspect(conn).get_columns(name)
                if c["name"] == "needs_reprocess"
            )
            assert not col["nullable"], name


def test_downgrade_removes_it_everywhere(migration, engine):
    eng, _ = engine
    with eng.begin() as conn:
        _run(migration, conn)
        _run(migration, conn, "downgrade")
        for name in migration.LABEL_TABLES:
            cols = {c["name"] for c in sa.inspect(conn).get_columns(name)}
            assert "needs_reprocess" not in cols, name

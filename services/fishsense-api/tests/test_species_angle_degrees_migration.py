"""The `fish_angle_degrees` migration on `specieslabel`.

A plain `add_column`, so what is worth asserting is not that it runs but what
it leaves behind. This column is the mirror image of `needs_reprocess`
(`c7e4a91f2d38`): there the NULL backfill was the bug, because a cohort
predicate read NULL as "skip"; here NULL is the whole point, since 0° is a
real answer in an angle sweep and must stay distinguishable from an unreviewed
row.

So the assertions run in the opposite direction — a pre-existing row must come
out of the upgrade NULL, and must *not* have been defaulted to 0.0.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa


@pytest.fixture
def migration():
    from fishsense_api.alembic.versions import (
        a1d47f2c9b83_add_fish_angle_degrees_to_species_label as mod,
    )

    return mod


@pytest.fixture
def engine(migration):
    """`specieslabel` shaped as it was before the column existed, carrying a
    row written before the upgrade."""
    eng = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    table = sa.Table(
        migration.TABLE,
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("image_id", sa.Integer),
        sa.Column("label_studio_project_id", sa.Integer),
        sa.Column("fish_angle_category", sa.String()),
    )
    metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(table.insert().values(image_id=1, fish_angle_category="x > 15°"))
    return eng


def _run(migration, engine, direction):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(migration, direction)()


def _columns(engine, table):
    return {c["name"] for c in sa.inspect(engine).get_columns(table)}


def test_upgrade_adds_the_column(migration, engine):
    assert migration.COLUMN not in _columns(engine, migration.TABLE)
    _run(migration, engine, "upgrade")
    assert migration.COLUMN in _columns(engine, migration.TABLE)


def test_pre_existing_row_is_null_not_zero(migration, engine):
    """The backfill must leave NULL. Defaulting to 0.0 would silently assert
    that every historical frame was reviewed and found to be at zero degrees —
    a real value in this sweep, so the lie would be unrecoverable."""
    _run(migration, engine, "upgrade")
    with engine.begin() as conn:
        value = conn.execute(
            sa.text(f"select {migration.COLUMN} from {migration.TABLE}")  # nosec
        ).scalar_one()
    assert value is None


def test_upgrade_preserves_the_existing_category(migration, engine):
    """The new column is additive — the labeler's category is untouched."""
    _run(migration, engine, "upgrade")
    with engine.begin() as conn:
        value = conn.execute(
            sa.text(f"select fish_angle_category from {migration.TABLE}")
        ).scalar_one()
    assert value == "x > 15°"


def test_downgrade_drops_the_column(migration, engine):
    _run(migration, engine, "upgrade")
    _run(migration, engine, "downgrade")
    assert migration.COLUMN not in _columns(engine, migration.TABLE)

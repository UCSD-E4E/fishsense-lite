"""The canonical-checksum index migration must never crash-loop the API.

`lifespan` runs `run_alembic_upgrade` on startup, so a migration that raises
takes fishsense-api down on deploy — and there is no staging tier to catch it
first. `uq_image_canonical_checksum` is a *unique* index over existing prod
data, which is exactly the kind of DDL that can fail on rows that predate the
invariant.

So the migration checks before it creates, and on finding violations logs and
returns rather than raising or repairing. Repair is an operator decision (which
copy is the real one is a judgement about the dives), not something a schema
migration should settle unattended.

These tests drive the migration against real sqlite, not mocks: the point is
whether the DDL and the guard behave against a database.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine

_VERSIONS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fishsense_api"
    / "alembic"
    / "versions"
)


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location(
        "canonical_index_migration",
        _VERSIONS / "a7f2c9e41b03_unique_canonical_image_per_checksum.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _image_table(metadata: sa.MetaData) -> sa.Table:
    return sa.Table(
        "image",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("path", sa.String(255), unique=True),
        sa.Column("taken_datetime", sa.DateTime(timezone=True)),
        sa.Column("checksum", sa.String(32)),
        sa.Column("is_canonical", sa.Boolean),
    )


@pytest.fixture
def engine(tmp_path):
    """File-backed sqlite: alembic's `op` needs a real connection, and the
    inspector must see committed DDL."""
    eng = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = sa.MetaData()
    table = _image_table(metadata)
    metadata.create_all(eng)
    yield eng, table
    eng.dispose()


def _seed(conn, table, rows):
    for image_id, checksum, canonical in rows:
        conn.execute(
            table.insert().values(
                id=image_id,
                path=f"d/{image_id}.ORF",
                taken_datetime=datetime(2024, 8, 21, tzinfo=timezone.utc),
                checksum=checksum,
                is_canonical=canonical,
            )
        )


def _run_upgrade(migration, conn):
    """Drive the migration body against a live connection.

    `alembic.op` is a module-level proxy bound to a MigrationContext, so the
    migration is run through a real (if minimal) alembic context rather than by
    monkeypatching `op`.
    """
    from alembic.migration import MigrationContext  # pylint: disable=import-outside-toplevel
    from alembic.operations import Operations  # pylint: disable=import-outside-toplevel

    with Operations.context(MigrationContext.configure(conn)):
        migration.upgrade()


def _indexes(conn) -> set[str]:
    return {ix["name"] for ix in sa.inspect(conn).get_indexes("image")}


def test_creates_the_index_when_existing_data_is_clean(migration, engine):
    eng, table = engine
    with eng.begin() as conn:
        _seed(
            conn,
            table,
            [
                (1, "a" * 32, True),
                (2, "a" * 32, False),   # duplicate content, correctly demoted
                (3, "b" * 32, True),
            ],
        )
        _run_upgrade(migration, conn)

        assert migration.INDEX_NAME in _indexes(conn)


def test_the_created_index_actually_rejects_a_second_canonical(migration, engine):
    """The index has to be *partial*. A plain unique index on `checksum` would
    also pass "was it created", while breaking the duplicate-frames case the
    pipeline depends on."""
    eng, table = engine
    with eng.begin() as conn:
        _seed(conn, table, [(1, "a" * 32, True)])
        _run_upgrade(migration, conn)

        # A second NON-canonical row with the same checksum is fine...
        _seed(conn, table, [(2, "a" * 32, False)])

        # ... a second canonical one is not.
        with pytest.raises(sa.exc.IntegrityError):
            _seed(conn, table, [(3, "a" * 32, True)])


def test_skips_without_raising_when_existing_data_violates_the_invariant(
    migration, engine, caplog
):
    """The crash-loop guard. Two canonical rows share a checksum, so the index
    cannot be created — the migration must log and return, leaving the schema
    untouched and the API able to start."""
    eng, table = engine
    with eng.begin() as conn:
        _seed(conn, table, [(1, "a" * 32, True), (2, "a" * 32, True)])

        with caplog.at_level("ERROR"):
            _run_upgrade(migration, conn)   # must not raise

        assert migration.INDEX_NAME not in _indexes(conn)
        assert "REFUSING" in caplog.text
        assert "a" * 32 in caplog.text      # names the offending checksum


def test_leaves_the_violating_rows_untouched(migration, engine):
    """It must not silently 'repair' prod data. Demoting the wrong copy would
    change which dive reads as canonical, and there is no staging tier to
    notice."""
    eng, table = engine
    with eng.begin() as conn:
        _seed(conn, table, [(1, "a" * 32, True), (2, "a" * 32, True)])

        _run_upgrade(migration, conn)

        still_canonical = conn.execute(
            sa.text("SELECT COUNT(*) FROM image WHERE is_canonical")
        ).scalar()
        assert still_canonical == 2


def test_is_idempotent_when_the_index_already_exists(migration, engine):
    eng, table = engine
    with eng.begin() as conn:
        _seed(conn, table, [(1, "a" * 32, True)])
        _run_upgrade(migration, conn)
        _run_upgrade(migration, conn)   # must not raise

        assert migration.INDEX_NAME in _indexes(conn)


def test_downgrade_removes_the_index_and_is_safe_when_absent(migration, engine):
    eng, table = engine
    with eng.begin() as conn:
        _seed(conn, table, [(1, "a" * 32, True)])
        _run_upgrade(migration, conn)

        from alembic.migration import MigrationContext  # pylint: disable=import-outside-toplevel
        from alembic.operations import Operations  # pylint: disable=import-outside-toplevel

        with Operations.context(MigrationContext.configure(conn)):
            migration.downgrade()
            assert migration.INDEX_NAME not in _indexes(conn)
            migration.downgrade()   # already gone — must not raise

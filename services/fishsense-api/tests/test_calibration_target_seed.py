"""The E4E checkerboard's seeded row, and the two ways it can be absent.

The table shipped empty on purpose — a calibration target without a measured
`square_size_m` can only produce confidently wrong lengths. This is that
measurement arriving, so what these pin down is that it arrives *everywhere*
and that it can be corrected:

  * the migration seeds it on an existing database;
  * `_seed_calibration_targets` seeds it on a fresh one, where
    `run_alembic_upgrade` STAMPS head and runs no migration at all — the
    bootstrap hole the views and `fishmodelreference` both fell into;
  * seeding is insert-only, so a re-measured board is not stamped back;
  * the row's `name` matches the species-taxonomy leaf, which is the join key
    the whole identification path turns on.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fishsense_shared import taxonomy

from fishsense_api import views

MODULE = "fishsense_api.alembic.versions.e07c31b9a4d2_seed_e4e_checkerboard_target"

E4E = "E4E Checkerboard"


def _row(name: str = E4E) -> dict:
    return next(t for t in views.KNOWN_CALIBRATION_TARGETS if t["name"] == name)


# ---------- the constant ----------


def test_the_e4e_board_is_the_measured_geometry():
    """14 x 10 INTERIOR corners (a 15 x 11 board), 42 mm pitch."""
    target = _row()
    assert (target["rows"], target["cols"]) == (10, 14)
    assert target["square_size_m"] == 0.042


def test_the_name_is_the_taxonomy_leaf():
    """The join key for the entire identification path.

    Species sync matches the `Calibration Targets` leaf against
    `CalibrationTarget.name`. If the two drift, the hook resolves nothing and
    every checkerboard dive silently stays uncalibrated — the same silence
    this stage was built to end.
    """
    assert _row()["name"] == taxonomy.CHECKERBOARD_NAME


def test_the_provenance_travels_with_the_number():
    """`square_size_m` alone cannot say how well it is known.

    Measured with a ruler across one square, so it resolves to about
    +-0.5 mm — roughly +-1.2% of scale, in the one direction reprojection
    residual provably cannot see. A reader who takes 0.042 as exact would
    misread every length that follows from it, and nothing downstream can
    flag that.
    """
    notes = _row()["notes"]
    assert "4.2" in notes
    assert "ruler" in notes.lower()
    assert "1.2%" in notes


def test_the_notes_forbid_back_solving_from_the_fish_models():
    """The known lengths are the validation set, never a calibration input.

    A scale bias they reveal is evidence to re-measure the board. A pitch
    fitted to them would make every future accuracy number self-confirming and
    destroy the only independent check the measurement pipeline has — so the
    warning has to travel with the row, where someone tempted to "improve" the
    number will actually read it.
    """
    notes = _row()["notes"].lower()
    assert "back-solve" in notes
    assert "validation" in notes


def test_the_grid_is_stated_as_interior_corners():
    """A 15 x 11 board has 14 x 10 interior corners, and confusing the two is
    a 7-10% scale error that would calibrate cleanly."""
    assert "interior corner" in _row()["notes"].lower()


# ---------- the migration ----------


def _upgrade(connection) -> None:
    import importlib

    module = importlib.import_module(MODULE)
    context = MigrationContext.configure(connection)
    with Operations.context(context):
        module.upgrade()


def _table(connection) -> None:
    connection.execute(
        sa.text(
            "CREATE TABLE calibrationtarget ("
            "id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE, "
            "rows INTEGER NOT NULL, cols INTEGER NOT NULL, "
            "square_size_m FLOAT NOT NULL, notes VARCHAR, created_at DATETIME)"
        )
    )


def _stored(connection) -> dict:
    return {
        r[0]: {"rows": r[1], "cols": r[2], "square_size_m": r[3], "notes": r[4]}
        for r in connection.execute(
            sa.text("SELECT name, rows, cols, square_size_m, notes FROM calibrationtarget")
        )
    }


def test_migration_seeds_the_board():
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _table(connection)

        _upgrade(connection)

        stored = _stored(connection)
        assert set(stored) == {t["name"] for t in views.KNOWN_CALIBRATION_TARGETS}
        assert stored[E4E]["square_size_m"] == 0.042
        assert (stored[E4E]["rows"], stored[E4E]["cols"]) == (10, 14)


def test_migration_is_insert_only():
    """A re-measured board must not be stamped back to the seeded value.

    Not hypothetical: 4.2 cm is a single-square reading, and the standing
    advice is to re-measure across many squares and divide. When someone does,
    their correction has to survive every subsequent deploy.
    """
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _table(connection)
        connection.execute(
            sa.text(
                "INSERT INTO calibrationtarget "
                "(name, rows, cols, square_size_m, notes) "
                "VALUES (:name, 10, 14, 0.04187, 'recalipered over 10 squares')"
            ),
            {"name": E4E},
        )

        _upgrade(connection)

        stored = _stored(connection)
        assert stored[E4E]["square_size_m"] == 0.04187
        assert stored[E4E]["notes"] == "recalipered over 10 squares"


def test_migration_is_idempotent():
    """`create_all` runs before alembic on every restart, and a re-applied
    migration must not raise on the unique `name`."""
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _table(connection)

        _upgrade(connection)
        _upgrade(connection)

        assert len(_stored(connection)) == len(views.KNOWN_CALIBRATION_TARGETS)


def test_downgrade_removes_only_the_seeded_rows():
    import importlib

    module = importlib.import_module(MODULE)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        _table(connection)
        connection.execute(
            sa.text(
                "INSERT INTO calibrationtarget "
                "(name, rows, cols, square_size_m) "
                "VALUES ('Someone elses board', 8, 11, 0.03)"
            )
        )
        _upgrade(connection)

        context = MigrationContext.configure(connection)
        with Operations.context(context):
            module.downgrade()

        assert set(_stored(connection)) == {"Someone elses board"}


# ---------- the fresh-database bootstrap ----------


async def _seed_against(tmp_path, monkeypatch, preexisting_sql=None, times=1):
    """Run `_seed_calibration_targets` against a real file-backed SQLite.

    A file rather than `:memory:` because the helper disposes the engine it
    creates, which would discard an in-memory database before the test could
    read it back.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import SQLModel

    from fishsense_api import database

    url = f"sqlite+aiosqlite:///{tmp_path}/seed.db"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        if preexisting_sql:
            await conn.execute(sa.text(preexisting_sql))
    await engine.dispose()

    monkeypatch.setattr(database, "pg_connection_string", lambda: url)
    for _ in range(times):
        await database._seed_calibration_targets()  # pylint: disable=protected-access

    reader = create_async_engine(url)
    async with reader.connect() as conn:
        rows = {
            r[0]: {"rows": r[1], "cols": r[2], "square_size_m": r[3]}
            for r in await conn.execute(
                sa.text("SELECT name, rows, cols, square_size_m FROM calibrationtarget")
            )
        }
    await reader.dispose()
    return rows


async def test_fresh_database_seeding_inserts_the_board(tmp_path, monkeypatch):
    """On a fresh DB `run_alembic_upgrade` STAMPS head rather than upgrading,
    so the migration above never runs and the table comes up empty — leaving
    every checkerboard dive uncalibrated with nothing saying why."""
    rows = await _seed_against(tmp_path, monkeypatch)

    assert set(rows) == {t["name"] for t in views.KNOWN_CALIBRATION_TARGETS}
    assert rows[E4E]["square_size_m"] == 0.042


async def test_fresh_seeding_is_insert_only_and_idempotent(tmp_path, monkeypatch):
    """It runs on every start, so it must be safe to repeat — and must never
    overwrite a re-measured pitch."""
    rows = await _seed_against(
        tmp_path,
        monkeypatch,
        preexisting_sql=(
            "INSERT INTO calibrationtarget (name, rows, cols, square_size_m) "
            f"VALUES ('{E4E}', 10, 14, 0.04187)"
        ),
        times=2,
    )

    assert rows[E4E]["square_size_m"] == 0.04187
    assert len(rows) == len(views.KNOWN_CALIBRATION_TARGETS)

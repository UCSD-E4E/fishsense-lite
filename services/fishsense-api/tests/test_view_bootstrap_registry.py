"""`views.ALL_VIEW_DDL` must list every view, or fresh environments lose it.

Views are raw SQL owned by migrations and are not part of
`SQLModel.metadata`. On a fresh database `run_alembic_upgrade` stamps head
rather than upgrading, so the migrations that create them never run — and the
stamped version means it never self-heals. `database._create_all_views` covers
that by replaying `ALL_VIEW_DDL`.

Which makes that tuple a registry with the same failure mode as
`controllers/__init__.py` and `database.py`'s model imports: forget to add
your entry and nothing complains, it just silently isn't there. This test is
the complaint.
"""

from __future__ import annotations

import re

from fishsense_api import views


def _declared_view_names() -> set[str]:
    """Every `*_VIEW_NAME` constant declared in views.py."""
    return {
        getattr(views, name)
        for name in dir(views)
        if name.endswith("_VIEW_NAME")
    }


def test_every_declared_view_is_in_the_bootstrap_registry():
    missing = _declared_view_names() - set(views.ALL_VIEW_NAMES)
    assert not missing, (
        f"{sorted(missing)} declared in views.py but absent from ALL_VIEW_NAMES; "
        "a fresh database would silently not have it"
    )


def test_the_registry_has_one_ddl_pair_per_name():
    assert len(views.ALL_VIEW_DDL) == len(views.ALL_VIEW_NAMES)


def test_every_registry_entry_drops_then_creates_the_same_view():
    """A mismatched pair would drop one view and create another, which reads
    as working right up until the wrong one goes missing."""
    for (drop_sql, create_sql), name in zip(
        views.ALL_VIEW_DDL, views.ALL_VIEW_NAMES, strict=True
    ):
        assert re.search(rf"DROP VIEW IF EXISTS {name}\b", drop_sql), (drop_sql, name)
        assert re.search(rf"CREATE VIEW {name}\b", create_sql), name


def test_registry_names_are_unique():
    assert len(set(views.ALL_VIEW_NAMES)) == len(views.ALL_VIEW_NAMES)


# ── seed data has the same bootstrap hole views had ───────────────────


async def _seed_against(tmp_path, monkeypatch, preexisting_sql=None, times=1):
    """Run `_seed_fish_model_references` against a real file-backed SQLite.

    A file rather than `:memory:` because the helper disposes the engine it
    creates, which would discard an in-memory database before the test could
    read it back.
    """
    import sqlalchemy as sa
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
        await database._seed_fish_model_references()  # pylint: disable=protected-access

    reader = create_async_engine(url)
    async with reader.connect() as conn:
        rows = {
            r[0]: {"length": r[1], "notes": r[2], "provisional": bool(r[3])}
            for r in await conn.execute(
                sa.text(
                    "SELECT name, known_length_m, notes, is_provisional "
                    "FROM fishmodelreference"
                )
            )
        }
    await reader.dispose()
    return rows


async def test_fresh_database_seeding_inserts_every_known_model(
    tmp_path, monkeypatch
):
    """On a fresh DB `run_alembic_upgrade` STAMPS head instead of upgrading, so
    every seed the migrations perform is skipped and `fishmodelreference` comes
    up empty — leaving `fish_model_measurement_accuracy` empty for every model.

    That is the same silent absence the Weasly Fish work set out to end,
    reintroduced through the bootstrap path. Views had a replay for it; seed
    rows did not.
    """
    rows = await _seed_against(tmp_path, monkeypatch)

    assert set(rows) == {m["name"] for m in views.KNOWN_FISH_MODELS}
    # Caliper notes must survive the bootstrap. Nothing is provisional: every
    # reference length is a real measurement.
    assert "58.69" in rows["Weasly Fish"]["notes"]
    assert not any(r["provisional"] for r in rows.values())


async def test_seeding_is_insert_only_and_idempotent(tmp_path, monkeypatch):
    """An operator who calipered a model and corrected its row must not have
    that stamped back to the seeded estimate on the next restart — and the
    helper runs on every start, so it must also be safe to repeat."""
    rows = await _seed_against(
        tmp_path,
        monkeypatch,
        preexisting_sql=(
            "INSERT INTO fishmodelreference "
            "(name, known_length_m, notes, is_provisional) "
            "VALUES ('Weasly Fish', 0.287, 'calipered', 0)"
        ),
        times=2,
    )

    assert rows["Weasly Fish"]["length"] == 0.287
    assert rows["Weasly Fish"]["notes"] == "calipered"
    assert rows["Weasly Fish"]["provisional"] is False

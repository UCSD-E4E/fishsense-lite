"""Tests for the auto-migrate on startup behavior.

The FastAPI lifespan applies pending alembic migrations before
serving traffic so a deploy that ships a new migration (e.g. the
`dive_pipeline_status` view) doesn't require a manual operator step.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fishsense_api import database


def _patch_run_alembic(has_alembic_version: bool):
    """Helper: mock the alembic deps + the fresh-vs-existing detector.

    Returns the (fake_command, fake_cfg) so the caller can assert on
    which alembic verb fired and the Config wiring.
    """

    async def _fake_check():
        return has_alembic_version

    fake_command = MagicMock()
    fake_config_cls = MagicMock()
    fake_cfg = MagicMock()
    fake_config_cls.return_value = fake_cfg

    return (
        fake_command,
        fake_cfg,
        patch.multiple(
            database,
            alembic_command=fake_command,
            AlembicConfig=fake_config_cls,
            _has_alembic_version_table=_fake_check,
        ),
    )


def _assert_script_location_set(fake_cfg):
    """The Config's `script_location` must point at the package's
    alembic directory so the migration scripts are found inside the
    installed wheel (where alembic.ini does NOT ship)."""
    set_calls = fake_cfg.set_main_option.call_args_list
    script_location_call = next(
        c for c in set_calls if c.args[0] == "script_location"
    )
    expected = str(Path(database.__file__).resolve().parent / "alembic")
    assert script_location_call.args[1] == expected


def test_run_alembic_upgrade_existing_db_invokes_upgrade_to_head():
    """When alembic_version exists, this is an existing DB — apply any
    new pending migrations via `alembic upgrade head`."""
    fake_command, fake_cfg, patcher = _patch_run_alembic(has_alembic_version=True)
    with patcher:
        database.run_alembic_upgrade()

    fake_command.upgrade.assert_called_once_with(fake_cfg, "head")
    fake_command.stamp.assert_not_called()
    _assert_script_location_set(fake_cfg)


def test_run_alembic_upgrade_fresh_db_stamps_head_without_running_ddl():
    """When alembic_version is absent, the lifespan's prior
    `create_all` has populated the full ORM-defined schema. Stamp
    head to mark the DB as fully migrated — running the historical
    migration tail on top would crash on every pre-existing column /
    table / constraint that `add_column` / `create_table` /
    `create_unique_constraint` ops try to (re-)add."""
    fake_command, fake_cfg, patcher = _patch_run_alembic(has_alembic_version=False)
    with patcher:
        database.run_alembic_upgrade()

    fake_command.stamp.assert_called_once_with(fake_cfg, "head")
    fake_command.upgrade.assert_not_called()
    _assert_script_location_set(fake_cfg)


@pytest.mark.asyncio
async def test_lifespan_runs_alembic_upgrade_after_setup(monkeypatch):
    """Wired-up regression: the FastAPI lifespan must invoke the
    upgrade so the deployed image catches up the schema on its own."""
    from fishsense_api import server  # pylint: disable=import-outside-toplevel

    fake_db = MagicMock()
    fake_engine_begin = MagicMock()
    fake_engine_begin.__aenter__ = MagicMock(
        return_value=__import__("asyncio").get_event_loop().create_future()
    )
    # Simpler: stub setup_database + dispose + init_database to no-ops, and
    # assert the upgrade-runner was awaited before yield.
    fake_db.engine.begin = MagicMock(return_value=_AsyncCM())
    fake_db.dispose = _AsyncNoop()

    upgrade_calls = []

    async def fake_to_thread(fn, *_args, **_kwargs):
        upgrade_calls.append(fn)

    monkeypatch.setattr(server, "setup_database", lambda: fake_db)
    monkeypatch.setattr(server.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(
        server, "run_alembic_upgrade", lambda: "called"
    )
    fake_db.init_database = _AsyncNoop()

    async with server.lifespan(None):
        pass

    assert server.run_alembic_upgrade in upgrade_calls, (
        "lifespan must call asyncio.to_thread(run_alembic_upgrade)"
    )


class _AsyncCM:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AsyncNoop:
    def __call__(self, *args, **kwargs):
        async def _coro():
            return None

        return _coro()

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


def test_run_alembic_upgrade_invokes_command_upgrade_to_head():
    """The function must end up calling `alembic.command.upgrade(cfg,
    "head")`. The Config's `script_location` must point at the
    package's alembic directory so the migration scripts are found
    inside the installed wheel (where alembic.ini does NOT ship)."""
    fake_command = MagicMock()
    fake_config_cls = MagicMock()
    fake_cfg = MagicMock()
    fake_config_cls.return_value = fake_cfg

    with patch.object(database, "alembic_command", fake_command), patch.object(
        database, "AlembicConfig", fake_config_cls
    ):
        database.run_alembic_upgrade()

    fake_command.upgrade.assert_called_once_with(fake_cfg, "head")
    set_calls = fake_cfg.set_main_option.call_args_list
    script_location_call = next(
        c for c in set_calls if c.args[0] == "script_location"
    )
    expected = str(
        Path(database.__file__).resolve().parent / "alembic"
    )
    assert script_location_call.args[1] == expected


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

"""Regression test for the headtailprediction migration's idempotency.

The FastAPI lifespan runs `SQLModel.metadata.create_all` *before*
`run_alembic_upgrade`. `headtailprediction` is a brand-new table in the ORM
model registry, so on an existing DB the deploy that ships the model creates
the table via `create_all` and *then* alembic reaches this migration's
`create_table`. Without a guard that raises `DuplicateTableError` and crashes
startup — the outage that took fishsense-api down once already.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_VERSIONS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fishsense_api"
    / "alembic"
    / "versions"
)


def _load(filename: str, mod_name: str):
    """Migration filenames start with a digit, so importlib.util is needed."""
    spec = importlib.util.spec_from_file_location(mod_name, _VERSIONS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migration():
    return _load(
        "e9c1a7b40d53_add_headtailprediction_table.py", "ht_create_migration"
    )


def test_upgrade_skips_when_table_already_exists(migration, monkeypatch):
    inspector = MagicMock()
    inspector.has_table.return_value = True
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: inspector)
    monkeypatch.setattr(migration.op, "get_bind", MagicMock())
    create_table = MagicMock()
    monkeypatch.setattr(migration.op, "create_table", create_table)
    monkeypatch.setattr(migration.op, "create_index", MagicMock())
    monkeypatch.setattr(migration.op, "f", lambda name: name)

    migration.upgrade()

    inspector.has_table.assert_called_once_with("headtailprediction")
    create_table.assert_not_called()


def test_upgrade_creates_when_table_absent(migration, monkeypatch):
    inspector = MagicMock()
    inspector.has_table.return_value = False
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: inspector)
    monkeypatch.setattr(migration.op, "get_bind", MagicMock())
    create_table = MagicMock()
    monkeypatch.setattr(migration.op, "create_table", create_table)
    monkeypatch.setattr(migration.op, "create_index", MagicMock())
    monkeypatch.setattr(migration.op, "f", lambda name: name)

    migration.upgrade()

    create_table.assert_called_once()
    assert create_table.call_args[0][0] == "headtailprediction"


def test_boolean_and_status_carry_server_defaults(migration, monkeypatch):
    """A NULL backfill would break both columns under three-valued logic.

    `WHERE NOT rejected_low_confidence` silently drops NULL rows, and a NULL
    `status` makes an abstention indistinguishable from a prediction. Both must
    be NOT NULL with a server default — the c7e4a91f2d38 lesson.
    """
    inspector = MagicMock()
    inspector.has_table.return_value = False
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: inspector)
    monkeypatch.setattr(migration.op, "get_bind", MagicMock())
    create_table = MagicMock()
    monkeypatch.setattr(migration.op, "create_table", create_table)
    monkeypatch.setattr(migration.op, "create_index", MagicMock())
    monkeypatch.setattr(migration.op, "f", lambda name: name)

    migration.upgrade()

    columns = {
        c.name: c
        for c in create_table.call_args[0][1:]
        if hasattr(c, "name") and hasattr(c, "nullable")
    }
    for name in ("status", "rejected_low_confidence"):
        assert name in columns, f"{name} column missing"
        assert columns[name].nullable is False, f"{name} must be NOT NULL"
        assert columns[name].server_default is not None, (
            f"{name} needs a server_default, or existing rows land NULL"
        )

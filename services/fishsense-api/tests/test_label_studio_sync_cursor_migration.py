"""Regression tests for the labelstudiosynccursor migration idempotency.

The FastAPI lifespan runs `SQLModel.metadata.create_all` *before*
`run_alembic_upgrade`. On a fresh-bootstrap deploy this means the
`labelstudiosynccursor` table already exists by the time the alembic
migration runs. The migration must skip its `create_table` op rather
than fail with `DuplicateTableError`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_migration_module():
    """Migration filenames start with a digit, so importlib.util is needed."""
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fishsense_api"
        / "alembic"
        / "versions"
        / "299428e39cb9_add_label_studio_sync_cursor.py"
    )
    spec = importlib.util.spec_from_file_location("ls_sync_cursor_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migration():
    return _load_migration_module()


def test_upgrade_skips_when_table_already_exists(migration, monkeypatch):
    """create_all already made the table — upgrade must be a no-op."""
    fake_inspector = MagicMock()
    fake_inspector.has_table.return_value = True

    fake_op = MagicMock()
    fake_op.get_bind.return_value = "bind"

    monkeypatch.setattr(migration, "op", fake_op)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: fake_inspector)

    migration.upgrade()

    fake_inspector.has_table.assert_called_once_with("labelstudiosynccursor")
    fake_op.create_table.assert_not_called()
    fake_op.create_index.assert_not_called()


def test_upgrade_creates_when_table_absent(migration, monkeypatch):
    """If the table somehow doesn't exist, the migration must still build it."""
    fake_inspector = MagicMock()
    fake_inspector.has_table.return_value = False

    fake_op = MagicMock()
    fake_op.get_bind.return_value = "bind"

    monkeypatch.setattr(migration, "op", fake_op)
    monkeypatch.setattr(migration.sa, "inspect", lambda _bind: fake_inspector)

    migration.upgrade()

    fake_op.create_table.assert_called_once()
    assert fake_op.create_index.call_count == 2

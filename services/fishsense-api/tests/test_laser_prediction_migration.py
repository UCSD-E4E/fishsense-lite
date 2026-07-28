"""Regression tests for the laserprediction migrations' idempotency.

The FastAPI lifespan runs `SQLModel.metadata.create_all` *before*
`run_alembic_upgrade`. `laserprediction` is a brand-new table, so on an
existing DB the deploy that ships the model creates the table via
`create_all` and *then* alembic's `upgrade head` reaches the
`create_table` / `add_column` ops for it. Without a guard those ops hit
`DuplicateTableError` / `DuplicateColumnError` and crash startup — which
is exactly the outage that took down fishsense-api. Both migrations must
skip their DDL when the table / columns already exist.
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
def create_migration():
    return _load("a7d21e4f0c93_add_laser_prediction_table.py", "lp_create_migration")


@pytest.fixture
def dims_migration():
    return _load("b8e3f1a09d24_add_dims_to_laser_prediction.py", "lp_dims_migration")


# ---------------- a7d21e4f0c93: create laserprediction ----------------


def test_create_upgrade_skips_when_table_exists(create_migration, monkeypatch):
    """create_all already made the table — upgrade must be a no-op."""
    fake_inspector = MagicMock()
    fake_inspector.has_table.return_value = True
    fake_op = MagicMock()
    fake_op.get_bind.return_value = "bind"

    monkeypatch.setattr(create_migration, "op", fake_op)
    monkeypatch.setattr(create_migration.sa, "inspect", lambda _b: fake_inspector)

    create_migration.upgrade()

    fake_inspector.has_table.assert_called_once_with("laserprediction")
    fake_op.create_table.assert_not_called()


def test_create_upgrade_creates_when_table_absent(create_migration, monkeypatch):
    """If the table somehow doesn't exist, the migration must still build it."""
    fake_inspector = MagicMock()
    fake_inspector.has_table.return_value = False
    fake_op = MagicMock()
    fake_op.get_bind.return_value = "bind"

    monkeypatch.setattr(create_migration, "op", fake_op)
    monkeypatch.setattr(create_migration.sa, "inspect", lambda _b: fake_inspector)

    create_migration.upgrade()

    fake_op.create_table.assert_called_once()


# ---------------- b8e3f1a09d24: add width/height ----------------


def _dims_setup(dims_migration, monkeypatch, existing_columns):
    fake_inspector = MagicMock()
    fake_inspector.get_columns.return_value = [
        {"name": name} for name in existing_columns
    ]
    fake_op = MagicMock()
    fake_op.get_bind.return_value = "bind"

    monkeypatch.setattr(dims_migration, "op", fake_op)
    monkeypatch.setattr(dims_migration.sa, "inspect", lambda _b: fake_inspector)
    return fake_op


def test_dims_upgrade_skips_when_both_columns_exist(dims_migration, monkeypatch):
    """create_all made the table with width+height already — no-op."""
    fake_op = _dims_setup(
        dims_migration,
        monkeypatch,
        ["id", "x", "y", "confidence", "width", "height", "created_at", "image_id"],
    )

    dims_migration.upgrade()

    fake_op.add_column.assert_not_called()


def test_dims_upgrade_adds_when_columns_absent(dims_migration, monkeypatch):
    """Table came from the create_table migration (no dims) — add both."""
    fake_op = _dims_setup(
        dims_migration,
        monkeypatch,
        ["id", "x", "y", "confidence", "created_at", "image_id"],
    )

    dims_migration.upgrade()

    added = {c.args[1].name for c in fake_op.add_column.call_args_list}
    assert added == {"width", "height"}


def test_dims_upgrade_adds_only_missing_column(dims_migration, monkeypatch):
    """Partial state (only width present) — add just the missing height."""
    fake_op = _dims_setup(
        dims_migration,
        monkeypatch,
        ["id", "x", "y", "confidence", "width", "created_at", "image_id"],
    )

    dims_migration.upgrade()

    added = {c.args[1].name for c in fake_op.add_column.call_args_list}
    assert added == {"height"}

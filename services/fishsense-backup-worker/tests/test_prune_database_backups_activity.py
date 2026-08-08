# pylint: disable=protected-access
"""Unit tests for the prune activity — the only code in this repo that
*deletes* backups.

The pure retention maths (`filenames_to_prune`) is covered in
`test_backup_naming_and_pruning.py`. This module is the I/O wrapper around it,
and it was at 0% coverage: nothing checked which directory it listed, or which
paths it handed to `delete`. Those are the two ways this can destroy data
rather than merely fail — deleting from the wrong folder, or deleting a
filename resolved against the wrong root.

CLAUDE.md: the nightly backup *is* the rollback mechanism for a system with no
staging tier. There is nothing behind it.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    """Dynaconf validates every Validator on first access; seed placeholders so
    importing the activity module doesn't fail first."""
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "https://nas.example.com:6021")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "u")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "p")
    monkeypatch.setenv("E4EFS_POSTGRES__HOST", "postgres")
    monkeypatch.setenv("E4EFS_POSTGRES__USERNAME", "backup")
    monkeypatch.setenv("E4EFS_POSTGRES__PASSWORD", "secret")
    yield


@pytest.fixture
def nas(monkeypatch):
    """A stand-in NAS whose `list_filenames`/`delete` calls are recorded."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    client = MagicMock()
    client.list_filenames.return_value = []
    monkeypatch.setattr(sut, "NasBackupClient", lambda **_kwargs: client)
    return client


def _prune(**kwargs):
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    return sut._prune(**kwargs)


# ── which directory it touches ────────────────────────────────────────


def test_lists_the_per_database_subdirectory_not_the_root(nas):
    """Backups are laid out `<root>/<db_name>/<file>`. Listing the root instead
    would return every database's files and prune across databases."""
    _prune(db_name="fishsense", nas_root_path="/backups", keep=3)

    nas.list_filenames.assert_called_once_with(folder_path="/backups/fishsense")


def test_a_trailing_slash_on_the_root_does_not_produce_a_double_slash(nas):
    """`//` resolves fine on some filesystems and 404s on FileStation, so a
    config value with a trailing slash must not change the path."""
    _prune(db_name="fishsense", nas_root_path="/backups/", keep=3)

    nas.list_filenames.assert_called_once_with(folder_path="/backups/fishsense")


# ── what it deletes ───────────────────────────────────────────────────


def test_deletes_each_pruned_file_by_full_path(nas, monkeypatch):
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    nas.list_filenames.return_value = ["a.dump", "b.dump", "c.dump"]
    monkeypatch.setattr(sut, "filenames_to_prune", lambda _files, keep: ["a.dump"])

    pruned = _prune(db_name="superset", nas_root_path="/backups", keep=2)

    nas.delete.assert_called_once_with(file_path="/backups/superset/a.dump")
    assert pruned == ["a.dump"]


def test_deletes_nothing_when_retention_says_nothing_to_prune(nas, monkeypatch):
    """The steady state on a fresh install. A wrapper that deleted on an empty
    prune list would clear the whole directory."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    nas.list_filenames.return_value = ["a.dump", "b.dump"]
    monkeypatch.setattr(sut, "filenames_to_prune", lambda _files, keep: [])

    assert _prune(db_name="fishsense", nas_root_path="/backups", keep=5) == []
    nas.delete.assert_not_called()


def test_deletes_only_what_the_retention_helper_returned(nas, monkeypatch):
    """The wrapper must not re-derive the set. Everything the helper didn't
    name has to survive."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    nas.list_filenames.return_value = [f"{i}.dump" for i in range(10)]
    monkeypatch.setattr(
        sut, "filenames_to_prune", lambda _files, keep: ["0.dump", "1.dump"]
    )

    _prune(db_name="fishsense", nas_root_path="/backups", keep=8)

    deleted = [c.kwargs["file_path"] for c in nas.delete.call_args_list]
    assert deleted == ["/backups/fishsense/0.dump", "/backups/fishsense/1.dump"]


@pytest.mark.usefixtures("nas")
def test_passes_keep_through_to_the_retention_helper(monkeypatch):
    """`keep` is the retention window. Dropping or defaulting it silently
    changes how much history survives."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    seen: List[int] = []

    def _spy(files, keep):  # pylint: disable=unused-argument
        seen.append(keep)
        return []

    monkeypatch.setattr(sut, "filenames_to_prune", _spy)

    _prune(db_name="fishsense", nas_root_path="/backups", keep=14)

    assert seen == [14]


def test_only_the_listed_files_are_considered(nas, monkeypatch):
    """The helper sees exactly what the NAS reported — no synthesised names."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    nas.list_filenames.return_value = ["x.dump", "y.dump"]
    seen: List[List[str]] = []

    def _spy(files, keep):  # pylint: disable=unused-argument
        seen.append(list(files))
        return []

    monkeypatch.setattr(sut, "filenames_to_prune", _spy)

    _prune(db_name="fishsense", nas_root_path="/backups", keep=3)

    assert seen == [["x.dump", "y.dump"]]


# ── the activity wrapper ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_activity_accepts_an_already_typed_payload(nas, monkeypatch):
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )
    from fishsense_backup_worker.workflows.backup_databases_workflow import (  # pylint: disable=import-outside-toplevel
        PruneDatabaseBackupsInput,
    )

    monkeypatch.setattr(sut, "filenames_to_prune", lambda _files, keep: [])

    await sut.prune_database_backups(
        PruneDatabaseBackupsInput(
            db_name="fishsense", nas_root_path="/backups", keep=3
        )
    )

    nas.list_filenames.assert_called_once_with(folder_path="/backups/fishsense")


@pytest.mark.asyncio
async def test_activity_validates_a_dict_payload(nas, monkeypatch):
    """Temporal hands the activity whatever the data converter produced; a
    plain dict must be coerced rather than attribute-errored."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        prune_database_backups as sut,
    )

    monkeypatch.setattr(sut, "filenames_to_prune", lambda _files, keep: [])

    await sut.prune_database_backups(
        {"db_name": "superset", "nas_root_path": "/backups", "keep": 4}
    )

    nas.list_filenames.assert_called_once_with(folder_path="/backups/superset")

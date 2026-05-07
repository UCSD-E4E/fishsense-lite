# pylint: disable=protected-access
"""Unit tests for NasBackupClient.

Pins the behavioral contract on the methods this worker uses
(`upload`, `list_filenames`, `delete`) independent of the underlying
NAS client implementation. The migration from synology-api to
synology-filestation (driven by the 2026-05-07 stage 2 incident in
the api-worker — see that worker's `nas.py` docstring) preserves
this surface; these tests are the regression guard if a future swap
ever breaks it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from synology_filestation import (
    AlreadyExists,
    NoSuchFile,
    SidNotFound,
)


def test_external_shape_preserved_for_activity_call_sites(monkeypatch):
    """Contract test for the public method shape of `NasBackupClient`.

    `pg_dump_database` and `prune_database_backups` call into this
    class using these exact keyword arguments. A future refactor that
    renames a method or drops a kwarg can't be caught at import time
    (call sites use `getattr` via `asyncio.to_thread`), so pin the
    shape here.
    """
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    fake = MagicMock(name="synology_filestation.Client")
    fake.list_dir.return_value = []
    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: fake)

    client = sut.NasBackupClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    # Asserting the calls don't raise TypeError is sufficient — we
    # don't care what the underlying client does, only that our
    # wrapper exposes these names with these kwarg names.
    client.upload(dest_dir="/foo/bar", src_file_path="/tmp/x.dump")
    client.list_filenames(folder_path="/foo/bar")
    client.delete(file_path="/foo/bar/old.dump")


def test_upload_propagates_underlying_failure(monkeypatch):
    """Behavioral invariant: an underlying upload failure must surface
    to the caller.

    The 2026-05-07 stage 2 incident in the api-worker was caused by
    silent error-swallowing in synology-api's `get_file`. A symmetric
    failure mode would exist here if the wrapper ever started
    swallowing typed exceptions from `Client.upload`. Pin the
    contract so it can't.
    """
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    fake = MagicMock(name="synology_filestation.Client")
    fake.upload.side_effect = SidNotFound("session expired")
    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: fake)

    client = sut.NasBackupClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    with pytest.raises(Exception):
        client.upload(dest_dir="/foo/bar", src_file_path="/tmp/x.dump")


def test_delete_propagates_underlying_failure(monkeypatch):
    """Same invariant for `delete` — the prune activity relies on
    deletion failures bubbling up so retention violations don't
    silently no-op.
    """
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    fake = MagicMock(name="synology_filestation.Client")
    fake.delete.side_effect = NoSuchFile("file not found")
    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: fake)

    client = sut.NasBackupClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    with pytest.raises(Exception):
        client.delete(file_path="/foo/bar/missing.dump")


def test_ensure_dir_treats_already_exists_as_success(monkeypatch):
    """`upload` calls `_ensure_dir` to create the destination folder
    before uploading. The new `synology_filestation.Client` raises
    typed `AlreadyExists` when the folder is already present; the
    wrapper must treat that as success rather than propagate (the
    folder-already-exists case is the common steady-state path).
    """
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    fake = MagicMock(name="synology_filestation.Client")
    fake.create_folder.side_effect = AlreadyExists("folder exists")
    fake.upload.return_value = None  # success
    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: fake)

    client = sut.NasBackupClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    # Should not raise — AlreadyExists during dir creation is a no-op.
    client.upload(dest_dir="/foo/bar", src_file_path="/tmp/x.dump")
    fake.upload.assert_called_once()


def test_list_filenames_returns_basenames(monkeypatch):
    """`list_dir` returns dicts with full paths in `name`; the wrapper
    must strip them down to bare basenames so the prune activity's
    string-matching logic works (`backup_naming.filenames_to_prune`
    operates on filenames, not absolute paths).
    """
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    fake = MagicMock(name="synology_filestation.Client")
    fake.list_dir.return_value = [
        {"path": "/backups/fishsense/2026-05-01.dump", "isdir": False, "size": 1024},
        {"path": "/backups/fishsense/2026-05-02.dump", "isdir": False, "size": 1024},
        {"path": "/backups/fishsense/old", "isdir": True, "size": 0},
    ]
    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: fake)

    client = sut.NasBackupClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    names = client.list_filenames(folder_path="/backups/fishsense")
    assert names == ["2026-05-01.dump", "2026-05-02.dump", "old"]

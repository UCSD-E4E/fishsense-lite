# pylint: disable=protected-access
"""Unit tests for `_dump_and_upload`.

Pins the regression for the 2026-05-03 incident: when the workflow
fans out per-DB activities in parallel, all three resolved their
rename target to the same `/tmp/<timestamp>.dump` path, raced on
`os.replace`, and the fastest-finishing activity's `finally`
cleanup deleted the file out from under the others' upload —
surfacing as `FileNotFoundError: File not found:
/tmp/2026-05-03T06-52-46Z.dump` on the slower siblings.

The fix is a per-activity `TemporaryDirectory`, so concurrent
invocations cannot share a path. These tests assert that.
"""

from __future__ import annotations

import os
import threading
from typing import List

import pytest


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    """Dynaconf eagerly validates every Validator; seed placeholders so
    importing the activity module doesn't fail before we get to the
    actual test logic."""
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "https://nas.example.com:6021")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "u")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "p")
    monkeypatch.setenv("E4EFS_POSTGRES__HOST", "postgres")
    monkeypatch.setenv("E4EFS_POSTGRES__USERNAME", "backup")
    monkeypatch.setenv("E4EFS_POSTGRES__PASSWORD", "secret")
    yield


def test_each_invocation_uses_an_isolated_tempdir(monkeypatch):
    """A single call must produce a path that's unique to its own
    tempdir, not a shared `/tmp` location. This is the static
    property that makes parallel callers safe."""
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        pg_dump_database as sut,
    )

    captured: List[str] = []

    def fake_pg_dump(*, output_path: str, **_kwargs):
        # Touch the file so the (mocked) upload sees a real file.
        with open(output_path, "wb") as fh:
            fh.write(b"")
        captured.append(output_path)

    class FakeNas:
        def __init__(self, **_kwargs):
            pass

        def upload(self, *, dest_dir, src_file_path):  # pylint: disable=unused-argument
            assert os.path.exists(src_file_path), (
                f"upload called with non-existent {src_file_path!r} — "
                "the tempdir lifecycle is wrong"
            )

    monkeypatch.setattr(sut, "run_pg_dump", fake_pg_dump)
    monkeypatch.setattr(sut, "NasBackupClient", FakeNas)

    sut._dump_and_upload(db_name="fishsense", nas_root_path="/backups")

    assert len(captured) == 1
    # Path must be inside a per-activity tempdir whose name includes
    # the db_name prefix — pins that an operator following a `lsof`
    # or `ls -la /tmp` can attribute the dir to a specific activity.
    parent = os.path.basename(os.path.dirname(captured[0]))
    assert parent.startswith("backup-fishsense-"), (
        f"tempdir parent {parent!r} doesn't carry the db_name prefix; "
        "operators won't be able to identify which activity owns a "
        "leftover dir if pg_dump ever wedges"
    )


def test_concurrent_calls_for_three_dbs_all_succeed(monkeypatch):
    """Regression for the 2026-05-03 incident: three concurrent
    `_dump_and_upload` invocations (one per DB) executing in
    parallel threads must each upload a real file. Pre-fix, all
    three resolved their rename target to the same
    `/tmp/<timestamp>.dump` and the first one's `finally` would
    delete the file before its siblings could upload, surfacing
    as `FileNotFoundError`.

    We test this by making `fake_pg_dump` block on a barrier so all
    three threads sit between `run_pg_dump` and `nas.upload`
    simultaneously — exactly the window the prior bug raced on.
    """
    from fishsense_backup_worker.activities import (  # pylint: disable=import-outside-toplevel
        pg_dump_database as sut,
    )

    upload_paths: List[str] = []
    upload_lock = threading.Lock()
    barrier = threading.Barrier(3)

    def fake_pg_dump(*, output_path: str, **_kwargs):
        with open(output_path, "wb") as fh:
            fh.write(b"\xff\xff")  # non-empty so a real file lands
        # Stall until all three sibling activities have written
        # their dump — exact window the prior bug raced on.
        barrier.wait(timeout=5.0)

    class FakeNas:
        def __init__(self, **_kwargs):
            pass

        def upload(self, *, dest_dir, src_file_path):  # pylint: disable=unused-argument
            assert os.path.exists(src_file_path), (
                f"upload called with non-existent {src_file_path!r} — "
                "a sibling's tempdir cleanup raced ahead"
            )
            with upload_lock:
                upload_paths.append(src_file_path)

    monkeypatch.setattr(sut, "run_pg_dump", fake_pg_dump)
    monkeypatch.setattr(sut, "NasBackupClient", FakeNas)

    threads = [
        threading.Thread(
            target=sut._dump_and_upload,
            kwargs={"db_name": db, "nas_root_path": "/backups"},
        )
        for db in ("fishsense", "superset", "temporal_db")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert all(not t.is_alive() for t in threads), "thread deadlocked"
    assert len(upload_paths) == 3
    # All three paths must be distinct — that's the property that
    # rules out the cross-activity race.
    assert len(set(upload_paths)) == 3, (
        f"two activities resolved to the same path: {upload_paths!r}"
    )
    # Each path must live in its own tempdir (different parent
    # directories).
    parents = {os.path.dirname(p) for p in upload_paths}
    assert len(parents) == 3, (
        f"two activities shared a tempdir parent: {parents!r}"
    )

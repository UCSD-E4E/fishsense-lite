"""`scan_and_register_images_activity` — the only ingest step that WRITES.

Everything before it reads: listing enumerates, preflight reads 1 MB headers and
decides. This one pulls whole files and creates `Image` rows, so its failure
modes are the expensive ones.

Three properties carry the weight:

  * **Skip without downloading.** A re-run over an already-ingested folder must
    cost nothing. Checking after the download would still be correct and would
    still move ~14.5 MB per frame across the NAS for no reason.
  * **Heartbeat-resume.** A batch is many whole-file downloads; a retry that
    restarted from zero would re-pull everything it had already registered.
  * **Reject, never default.** A frame with no readable timestamp is refused.
    Stage-1 clustering is pure timestamp arithmetic, so a fabricated value
    corrupts it silently — and `finalize` refuses to promote a dive with any
    rejection, so a bad frame stops the dive rather than entering it.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

# pylint: disable=import-error
# `tests` is not a unique package name in this workspace — see test_exif.py.
from ._tiff_builder import build_orf

DIVE = 412
CAMERA = 7
FOLDER = "2024.06.20.REEF/082929_FishModels_FSL07"


def _orf(date_time="2024:08:21 08:56:51") -> bytes:
    """Padded past one hash chunk so the streaming read is exercised."""
    return build_orf(date_time=date_time, serial_number="BJ6C67989") + b"\0" * 40_000


def _fs(existing_paths=()):
    """SDK stub. `images.get(dive_id=...)` is how existing paths are discovered
    — one call per batch rather than one lookup per frame."""
    fs = MagicMock()
    rows = []
    for p in existing_paths:
        row = MagicMock()
        row.path = p
        # Real rows carry a timestamp, and skipped frames must still feed the
        # batch max — see test_a_fully_ingested_rerun_still_reports_the_max.
        row.taken_datetime = datetime(2024, 8, 21, 7, 0, 0, tzinfo=timezone.utc)
        rows.append(row)
    fs.images.get = AsyncMock(return_value=rows)
    fs.images.post = AsyncMock(side_effect=lambda *a, **kw: 999)
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=False)
    return fs


def _nas(contents: dict[str, bytes], errors: dict[str, int] | None = None):
    nas = MagicMock()
    nas.downloaded = []

    def _download_to(*, src_path: str, dest_dir: str):
        name = src_path.rsplit("/", 1)[-1]
        nas.downloaded.append(name)
        code = (errors or {}).get(name)
        if code is not None:
            from synology_filestation import DSMError

            raise DSMError(f"Synology API error {code}")
        Path(dest_dir, name).write_bytes(contents[name])

    nas.download_to.side_effect = _download_to
    return nas


async def _run(paths, contents, monkeypatch, *, fs=None, errors=None, env=None):
    from fishsense_api_workflow_worker.activities import (
        scan_and_register_images_activity as sut,
    )

    nas = _nas(contents, errors)
    monkeypatch.setattr(sut, "build_nas_client", lambda: nas)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs or _fs())
    result = await (env or ActivityEnvironment()).run(
        sut.scan_and_register_images_activity,
        DIVE,
        [f"{FOLDER}/{p}" for p in paths],
        CAMERA,
    )
    return result, nas


# ── registering ───────────────────────────────────────────────────────


async def test_registers_a_frame_with_its_checksum_and_timestamp(monkeypatch):
    data = _orf()
    fs = _fs()

    result, _ = await _run(["A.ORF"], {"A.ORF": data}, monkeypatch, fs=fs)

    assert result.registered == 1
    assert not result.rejected
    (_dive, image), kwargs = fs.images.post.call_args
    assert _dive == DIVE
    assert image.checksum == hashlib.md5(data).hexdigest()
    assert image.taken_datetime == datetime(
        2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc
    )
    assert image.path == f"{FOLDER}/A.ORF"
    assert image.camera_id == CAMERA
    # The server computes canonicality; sending it would mark every duplicate
    # frame canonical and destroy the distinction #555 gates on.
    assert not kwargs.get("set_canonical")


async def test_the_checksum_is_md5_of_the_whole_file(monkeypatch):
    """The convention every migrated row follows. A reader that hashed only the
    header would agree with itself forever and disagree with all ~131k rows —
    and duplicate detection would silently report zero overlap."""
    data = _orf()
    fs = _fs()

    await _run(["A.ORF"], {"A.ORF": data}, monkeypatch, fs=fs)

    (_d, image), _k = fs.images.post.call_args
    assert image.checksum == hashlib.md5(data).hexdigest()


async def test_reports_the_max_timestamp_for_the_dive(monkeypatch):
    """`Dive.dive_datetime` is the MAX of its frames — how every existing dive
    row was derived. `finalize` needs it and only this step reads the EXIF."""
    early = _orf("2024:08:21 08:00:00")
    late = _orf("2024:08:21 09:30:00")

    result, _ = await _run(
        ["A.ORF", "B.ORF"], {"A.ORF": early, "B.ORF": late}, monkeypatch
    )

    assert result.max_taken_datetime == datetime(
        2024, 8, 21, 9, 30, 0, tzinfo=timezone.utc
    )


# ── skipping ──────────────────────────────────────────────────────────


async def test_an_already_registered_path_is_skipped_without_downloading(
    monkeypatch,
):
    """The property that makes a re-run cheap. Checking after the download
    would be just as correct and would still move ~14.5 MB per frame."""
    data = _orf()
    fs = _fs(existing_paths=[f"{FOLDER}/A.ORF"])

    result, nas = await _run(
        ["A.ORF", "B.ORF"], {"A.ORF": data, "B.ORF": data}, monkeypatch, fs=fs
    )

    assert result.skipped_existing == 1
    assert result.registered == 1
    assert nas.downloaded == ["B.ORF"]


async def test_a_fully_ingested_batch_downloads_nothing(monkeypatch):
    data = _orf()
    fs = _fs(existing_paths=[f"{FOLDER}/A.ORF", f"{FOLDER}/B.ORF"])

    result, nas = await _run(
        ["A.ORF", "B.ORF"], {"A.ORF": data, "B.ORF": data}, monkeypatch, fs=fs
    )

    assert result.skipped_existing == 2
    assert result.registered == 0
    assert not nas.downloaded
    assert fs.images.post.await_count == 0


# ── rejecting ─────────────────────────────────────────────────────────


async def test_a_frame_with_no_readable_timestamp_is_rejected_not_defaulted(
    monkeypatch,
):
    """Stage-1 clustering is pure timestamp arithmetic. `finalize` refuses to
    promote a dive with any rejection, so this stops the dive rather than
    quietly seeding a frame that will cluster wrongly forever."""
    blind = build_orf(date_time=None, date_time_original=None) + b"\0" * 40_000
    fs = _fs()

    result, _ = await _run(["A.ORF"], {"A.ORF": blind}, monkeypatch, fs=fs)

    assert result.registered == 0
    assert [r.path for r in result.rejected] == [f"{FOLDER}/A.ORF"]
    assert fs.images.post.await_count == 0


async def test_one_rejected_frame_does_not_stop_the_others(monkeypatch):
    """The batch reports everything it saw; `finalize` decides. Aborting here
    would hide how many frames are affected behind whichever failed first."""
    good, blind = _orf(), build_orf(date_time=None, date_time_original=None)

    result, _ = await _run(
        ["A.ORF", "B.ORF", "C.ORF"],
        {"A.ORF": good, "B.ORF": blind + b"\0" * 40_000, "C.ORF": good},
        monkeypatch,
    )

    assert result.registered == 2
    assert len(result.rejected) == 1


# ── NAS failure classification ────────────────────────────────────────


async def test_a_missing_file_fails_non_retryably(monkeypatch):
    """Synology 408 is "no such file" — waiting cannot fix a path that isn't
    there, so Temporal must not burn its retry budget. Unlike verification,
    ingest is trying to DO something: a missing frame is a failure, not a
    finding."""
    with pytest.raises(ApplicationError) as excinfo:
        await _run(["A.ORF"], {}, monkeypatch, errors={"A.ORF": 408})

    assert excinfo.value.non_retryable


async def test_a_transient_nas_error_propagates_for_temporal_to_retry(monkeypatch):
    """502 is the shared backend having a moment — routine and self-healing. It
    must reach the bounded jittered policy rather than becoming permanent."""
    from synology_filestation import DSMError

    with pytest.raises(DSMError):
        await _run(["A.ORF"], {}, monkeypatch, errors={"A.ORF": 502})


# ── heartbeat-resume ──────────────────────────────────────────────────


async def test_heartbeats_the_index_of_each_frame(monkeypatch):
    """A batch is many whole-file downloads. Without a per-frame heartbeat a
    retry restarts from zero and re-pulls everything already registered."""
    beats = []
    env = ActivityEnvironment()
    env.on_heartbeat = beats.append
    data = _orf()

    await _run(
        ["A.ORF", "B.ORF"],
        {"A.ORF": data, "B.ORF": data},
        monkeypatch,
        env=env,
    )

    assert beats == [0, 1]


async def test_resume_is_db_backed_not_heartbeat_backed(monkeypatch):
    """A retry must skip what was actually PERSISTED, not what a heartbeat
    index claims.

    Skipping frames below the heartbeat index would drop their outcomes from
    `BatchResult` — so a transient 502 mid-batch could erase an earlier no-EXIF
    rejection, and `finalize` would then see `rejected == 0` and promote a dive
    with a missing frame. That is precisely the guard `finalize` exists to be.

    Here a prior attempt registered A but rejected B (no EXIF). On retry, even
    with a heartbeat index past both: A is skipped because the DB says so, and
    B is re-read and re-rejected rather than silently forgotten.
    """
    import dataclasses

    good = _orf()
    blind = build_orf(date_time=None, date_time_original=None) + b"\0" * 40_000
    env = ActivityEnvironment()
    env.info = dataclasses.replace(env.info, heartbeat_details=[2])
    fs = _fs(existing_paths=[f"{FOLDER}/A.ORF"])

    result, nas = await _run(
        ["A.ORF", "B.ORF", "C.ORF"],
        {"A.ORF": good, "B.ORF": blind, "C.ORF": good},
        monkeypatch,
        fs=fs,
        env=env,
    )

    assert nas.downloaded == ["B.ORF", "C.ORF"]
    assert result.skipped_existing == 1
    assert result.registered == 1
    assert [r.path for r in result.rejected] == [f"{FOLDER}/B.ORF"]


# ── read-only where it must be ────────────────────────────────────────


def test_the_module_never_writes_to_the_nas():
    """Mirrors the cleanup activity's tripwire. The api-worker's NAS access is
    read-only by policy; ingest downloads and must never upload or delete."""
    import inspect

    from fishsense_api_workflow_worker.activities import (
        scan_and_register_images_activity as sut,
    )

    source = inspect.getsource(sut)
    for forbidden in (".upload(", ".delete(", "upload_bytes", "create_folder"):
        assert forbidden not in source, f"NAS must stay read-only: {forbidden}"


async def test_a_fully_ingested_rerun_still_reports_the_max_timestamp(monkeypatch):
    """`Dive.dive_datetime` is the MAX over the dive, and `finalize` gets it
    from here. Counting only newly-registered frames returned None for a
    fully-ingested batch, which would leave `dive_datetime` unset on any
    re-run — and something too early whenever a batch was partly skipped."""
    data = _orf()
    fs = _fs(existing_paths=[f"{FOLDER}/A.ORF", f"{FOLDER}/B.ORF"])

    result, nas = await _run(
        ["A.ORF", "B.ORF"], {"A.ORF": data, "B.ORF": data}, monkeypatch, fs=fs
    )

    assert not nas.downloaded
    assert result.registered == 0
    assert result.max_taken_datetime == datetime(
        2024, 8, 21, 7, 0, 0, tzinfo=timezone.utc
    )


async def test_a_skipped_frame_can_hold_the_batch_max(monkeypatch):
    """Mixed batch: the existing row is later than the new one, so the max has
    to come from the frame that was never downloaded."""
    early = _orf("2024:08:21 06:00:00")
    fs = _fs(existing_paths=[f"{FOLDER}/A.ORF"])

    result, _ = await _run(
        ["A.ORF", "B.ORF"], {"A.ORF": early, "B.ORF": early}, monkeypatch, fs=fs
    )

    assert result.registered == 1
    assert result.max_taken_datetime == datetime(
        2024, 8, 21, 7, 0, 0, tzinfo=timezone.utc
    )

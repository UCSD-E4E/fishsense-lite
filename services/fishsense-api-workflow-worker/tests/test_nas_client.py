# pylint: disable=protected-access
"""Unit tests for the api-worker's NasClient.

Pins the 2026-05-07 stage 2 incident: DSM returned 200 OK with a
JSON-error body after the staging activity's idle session timed out
(~30 min default DSM SID TTL). The previous synology-api 0.8.x
`get_file(mode='download')` implementation streamed the JSON-error
body to disk and returned a tuple instead of raising; the surrounding
wrapper only caught `FileStationError` exceptions, so the corruption
slipped through silently. Callers (`stage_raw_bytes_for_dive_activity`)
read the JSON bytes via `read_bytes()` and uploaded them to the
file-exchange as `.ORF` content — every subsequent stage-2
`preprocess_species_image` activity received those JSON bytes, failed
`rawpy.imread` with `LibRawIOError: Input/output error`, and Temporal
retried 60+ times before the workflow timed out.

The migration to `synology-filestation` (`Client`) eliminates the
underlying-library footgun: it raises typed exceptions on JSON errors
and writes downloads atomically. The test below pins the *behavioral
contract* on `NasClient.download_to` so any future implementation
swap (or accidental regression) re-trips the same alarm before
corrupt content can reach the file-exchange.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

import pytest
from synology_filestation import SidNotFound


def test_external_shape_preserved_for_activity_call_sites(monkeypatch):
    """Contract test for the public method shape of `NasClient`.

    `stage_raw_bytes_for_dive_activity`, `stage_slate_pdf_activity`,
    `archive_processed_jpegs_to_nas_activity`, and
    `cleanup_raw_bytes_for_dive_activity` all call into NasClient
    using these exact keyword arguments. A future refactor that
    renames a method or drops a kwarg can't be caught at import time
    (call sites use `getattr` via `asyncio.to_thread` partials), so
    pin the shape here.

    `NasDownloadClient` must remain an alias for `NasClient` —
    several activities import the narrow alias for documentation
    intent.
    """
    from fishsense_api_workflow_worker import nas as sut  # pylint: disable=import-outside-toplevel

    assert sut.NasDownloadClient is sut.NasClient

    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: MagicMock())
    client = sut.NasClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    # All three call shapes use keyword args. Asserting the calls don't
    # raise TypeError is sufficient — we don't care what the underlying
    # client does, only that our wrapper exposes these names.
    client.download_to(src_path="/foo", dest_dir="/tmp")
    client.upload(dest_dir="/foo", src_file_path="/tmp/bar")
    client.exists(file_path="/foo")


def test_download_to_raises_when_underlying_client_signals_failure(
    monkeypatch, tmp_path
):
    """Behavioral contract regression for the 2026-05-07 incident.

    When the underlying NAS client signals a download failure (today:
    by raising one of the `synology_filestation` typed exceptions on a
    DSM JSON-error response), `NasClient.download_to` MUST surface
    that failure to its caller AND MUST NOT leave any partial /
    JSON-error content visible at the destination path. Previously
    the wrapper silently swallowed tuple-returns from synology-api,
    leaving DSM JSON-error bodies on disk for the activity to read
    back as `.ORF` content.
    """
    from fishsense_api_workflow_worker import nas as sut  # pylint: disable=import-outside-toplevel

    src_path = "/share/data/2024.06.20.REEF/img.ORF"

    # Mock the boundary: NasClient delegates to synology_filestation's
    # Client. We simulate the new client's correct failure mode —
    # `download_to` raises a `SidNotFound` and (because of its atomic
    # `<local>.part`+rename semantics) leaves no file at the
    # destination.
    fake_fs = MagicMock(name="synology_filestation.Client")
    fake_fs.download_to.side_effect = SidNotFound("session expired")
    monkeypatch.setattr(sut.Client, "login", lambda *a, **kw: fake_fs)

    client = sut.NasClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    dest_dir = tmp_path / "stage"
    dest_dir.mkdir()

    # Property 1: download_to surfaces the failure to the caller.
    with pytest.raises(Exception):
        client.download_to(src_path=src_path, dest_dir=str(dest_dir))

    # Property 2: the destination doesn't contain JSON masquerading as
    # raw image bytes. (Atomic-rename means it should be absent
    # entirely; the assertion is broader to allow for a future client
    # whose semantic is "leave a valid file or none at all.")
    leftover = dest_dir / pathlib.Path(src_path).name
    if leftover.exists():
        contents = leftover.read_bytes()
        assert b'"success"' not in contents and b'"error"' not in contents, (
            "download_to left a DSM-shaped JSON-error body at the "
            "destination; the caller will read those bytes via "
            "read_bytes() and upload them to the file-exchange as `.ORF`, "
            "reproducing the 2026-05-07 stage 2 corruption."
        )

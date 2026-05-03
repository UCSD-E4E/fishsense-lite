# pylint: disable=unused-argument
"""Unit tests for stage_raw_bytes_for_dive_activity.

Pins down:
  1. HEAD-skips already-staged checksums (no NAS download, no PUT).
  2. Newly-staged images: NAS download → file-exchange PUT.
  3. Images without `path` or `checksum` get counted as `no_path`
     and don't crash.
  4. Returns the per-dive summary so the parent workflow can log
     counts.
  5. Share-relative `image.path` values (the DB convention) get
     `e4e_nas.raw_root_path` prepended before being handed to
     FileStation; absolute paths pass through unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    stage_raw_bytes_for_dive_activity as sut,
)


def _image(image_id: int, *, path: str = "/share/dive/IMG.ORF",
           checksum: str = "abc") -> Image:
    return Image(
        id=image_id,
        path=path,
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=checksum,
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )


def _make_fs(images: List[Image]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    return fs


def _make_nas() -> MagicMock:
    nas = MagicMock()
    # download_to writes a file to dest_dir; simulate with a side_effect.
    nas.download_to = MagicMock()
    return nas


def _patch_routes(monkeypatch, head_results: dict[str, bool]):
    """Mock httpx HEAD/PUT against the file-exchange.

    `head_results[checksum]` controls what HEAD returns for that
    checksum. Any checksum not in the map defaults to 404.
    """
    head_calls: List[str] = []
    put_calls: List[tuple] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "HEAD":
            checksum = path.rsplit("/", 1)[-1].removesuffix(".ORF")
            head_calls.append(checksum)
            present = head_results.get(checksum, False)
            return httpx.Response(200 if present else 404)
        if request.method == "PUT":
            checksum = path.rsplit("/", 1)[-1].removesuffix(".ORF")
            put_calls.append((checksum, request.content))
            return httpx.Response(201)
        return httpx.Response(405)

    transport = httpx.MockTransport(_mock_handler)

    real_async_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(sut.httpx, "AsyncClient", _client_factory)
    return head_calls, put_calls


def _patch_tempfile_read(monkeypatch, payload: bytes):
    """Make every Path(...).read_bytes() return `payload` so the test
    doesn't need to actually drop a real file on disk."""
    real_path = sut.Path

    class _PathStub(real_path):  # type: ignore[misc]
        def read_bytes(self):  # type: ignore[override]
            return payload

    monkeypatch.setattr(sut, "Path", _PathStub)


@pytest.mark.asyncio
async def test_skips_already_staged_checksums_no_nas_download(monkeypatch):
    images = [_image(1, checksum="aaa"), _image(2, checksum="bbb")]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    head_calls, put_calls = _patch_routes(
        monkeypatch, head_results={"aaa": True, "bbb": True}
    )

    result = await ActivityEnvironment().run(
        sut.stage_raw_bytes_for_dive_activity, 42
    )

    assert result.staged == 0
    assert result.skipped_already_present == 2
    assert result.no_path == 0
    nas.download_to.assert_not_called()
    assert not put_calls
    assert set(head_calls) == {"aaa", "bbb"}


@pytest.mark.asyncio
async def test_stages_new_checksums_via_nas_download_then_put(monkeypatch):
    images = [_image(1, checksum="aaa")]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"raw-bytes")

    _, put_calls = _patch_routes(monkeypatch, head_results={})

    result = await ActivityEnvironment().run(
        sut.stage_raw_bytes_for_dive_activity, 42
    )

    assert result.staged == 1
    assert result.skipped_already_present == 0
    assert nas.download_to.call_count == 1
    nas_call = nas.download_to.call_args
    assert nas_call.kwargs["src_path"] == "/share/dive/IMG.ORF"
    assert len(put_calls) == 1
    assert put_calls[0][0] == "aaa"


@pytest.mark.asyncio
async def test_counts_no_path_images_without_crashing(monkeypatch):
    images = [
        _image(1, path="", checksum="aaa"),
        _image(2, path="/x.ORF", checksum=""),
        _image(3, path="/y.ORF", checksum="ccc"),
    ]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"raw")

    _patch_routes(monkeypatch, head_results={})

    result = await ActivityEnvironment().run(
        sut.stage_raw_bytes_for_dive_activity, 42
    )

    assert result.no_path == 2
    assert result.staged == 1


@pytest.mark.asyncio
async def test_returns_zeros_when_dive_has_no_images(monkeypatch):
    fs = _make_fs([])
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_routes(monkeypatch, head_results={})

    result = await ActivityEnvironment().run(
        sut.stage_raw_bytes_for_dive_activity, 42
    )

    assert result.staged == 0
    assert result.skipped_already_present == 0
    assert result.no_path == 0
    nas.download_to.assert_not_called()


def test_resolve_nas_path_prepends_root_to_relative_paths(monkeypatch):
    """Share-relative paths (the DB convention) get the configured root
    prepended; absolute paths are returned unchanged."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    assert (
        sut._resolve_nas_path(  # pylint: disable=protected-access
            "2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF"
        )
        == "/fishsense_data/REEF/data/2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF"
    )
    # Absolute path: passthrough so an operator override or a future
    # path migration isn't double-prefixed.
    assert (
        sut._resolve_nas_path("/already/absolute/file.ORF")  # pylint: disable=protected-access
        == "/already/absolute/file.ORF"
    )
    # Trailing slash on root is normalized.
    monkeypatch.setenv("E4EFS_E4E_NAS__RAW_ROOT_PATH", "/foo/bar/")
    cfg.settings.reload()
    assert (
        sut._resolve_nas_path("rel/path.ORF")  # pylint: disable=protected-access
        == "/foo/bar/rel/path.ORF"
    )


@pytest.mark.asyncio
async def test_relative_image_paths_get_prefixed_before_nas_download(monkeypatch):
    """End-to-end: a share-relative `image.path` is rewritten with the
    root prefix before reaching the NAS client. Mirrors the prod path
    shape — DB stores `2024.06.20.REEF/.../P8290052.ORF`, NAS expects
    `/fishsense_data/REEF/data/2024.06.20.REEF/.../P8290052.ORF`."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    images = [
        _image(
            1,
            path="2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF",
            checksum="aaa",
        )
    ]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"raw-bytes")
    _, put_calls = _patch_routes(monkeypatch, head_results={})

    result = await ActivityEnvironment().run(
        sut.stage_raw_bytes_for_dive_activity, 42
    )

    assert result.staged == 1
    nas.download_to.assert_called_once()
    assert nas.download_to.call_args.kwargs["src_path"] == (
        "/fishsense_data/REEF/data/2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF"
    )
    # File-exchange PUT still keys on checksum, not the rewritten path.
    assert len(put_calls) == 1
    assert put_calls[0][0] == "aaa"

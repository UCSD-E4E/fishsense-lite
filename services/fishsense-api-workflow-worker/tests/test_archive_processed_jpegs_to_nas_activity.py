# pylint: disable=unused-argument
"""Unit tests for archive_processed_jpegs_to_nas_activity.

Pins down:
  1. NAS-skip when JPEG is already at the per-dive path (no GET, no upload).
  2. file-exchange 404 → counted as skipped_no_jpeg, no upload.
  3. Otherwise: GET from file-exchange → write to tempdir → NAS upload.
  4. NAS upload directory is `{root}/{workflow}/{dive_id}` per the
     agreed-on layout.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    archive_processed_jpegs_to_nas_activity as sut,
)


def _image(image_id: int, *, checksum: str = "abc") -> Image:
    return Image(
        id=image_id,
        path="/share/x.ORF",
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


def _patch_routes(monkeypatch, get_results: Dict[str, bytes | int]):
    """`get_results[checksum]` controls the /api/v1/exchange/{folder}/...
    GET response. bytes => 200 with body; int => that status code."""
    get_calls: List[str] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            checksum = request.url.path.rsplit("/", 1)[-1].removesuffix(".JPG")
            get_calls.append(checksum)
            result = get_results.get(checksum, 404)
            if isinstance(result, int):
                return httpx.Response(result)
            return httpx.Response(200, content=result)
        return httpx.Response(405)

    transport = httpx.MockTransport(_mock_handler)
    real_async_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(sut.httpx, "AsyncClient", _client_factory)
    return get_calls


@pytest.mark.asyncio
async def test_skips_already_archived_jpegs(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa")])
    nas = MagicMock()
    nas.exists = MagicMock(return_value=True)
    nas.upload = MagicMock()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    get_calls = _patch_routes(monkeypatch, get_results={})

    result = await ActivityEnvironment().run(
        sut.archive_processed_jpegs_to_nas_activity,
        42,
        "preprocess_jpeg",
        "laser",
    )

    assert result.archived == 0
    assert result.skipped_already_on_nas == 1
    assert result.skipped_no_jpeg == 0
    nas.upload.assert_not_called()
    assert not get_calls


@pytest.mark.asyncio
async def test_counts_no_jpeg_when_file_exchange_returns_404(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa")])
    nas = MagicMock()
    nas.exists = MagicMock(return_value=False)
    nas.upload = MagicMock()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_routes(monkeypatch, get_results={"aaa": 404})

    result = await ActivityEnvironment().run(
        sut.archive_processed_jpegs_to_nas_activity,
        42,
        "preprocess_jpeg",
        "laser",
    )

    assert result.archived == 0
    assert result.skipped_no_jpeg == 1
    nas.upload.assert_not_called()


@pytest.mark.asyncio
async def test_archives_via_get_then_nas_upload_with_correct_path(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa")])
    nas = MagicMock()
    nas.exists = MagicMock(return_value=False)
    nas.upload = MagicMock()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    get_calls = _patch_routes(
        monkeypatch, get_results={"aaa": b"jpeg-bytes"}
    )

    result = await ActivityEnvironment().run(
        sut.archive_processed_jpegs_to_nas_activity,
        42,
        "preprocess_jpeg",
        "laser",
    )

    assert result.archived == 1
    assert get_calls == ["aaa"]
    nas.upload.assert_called_once()
    upload_kwargs = nas.upload.call_args.kwargs
    # Path layout: {nas_root}/{workflow}/{dive_id} where the configured
    # default root is /fishsense_process_work/processed_jpegs.
    assert upload_kwargs["dest_dir"] == (
        "/fishsense_process_work/processed_jpegs/laser/42"
    )
    assert upload_kwargs["src_file_path"].endswith("aaa.JPG")


@pytest.mark.asyncio
async def test_returns_zeros_for_empty_dive(monkeypatch):
    fs = _make_fs([])
    nas = MagicMock()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_routes(monkeypatch, get_results={})

    result = await ActivityEnvironment().run(
        sut.archive_processed_jpegs_to_nas_activity,
        42,
        "preprocess_jpeg",
        "laser",
    )

    assert result.archived == 0
    assert result.skipped_already_on_nas == 0
    assert result.skipped_no_jpeg == 0

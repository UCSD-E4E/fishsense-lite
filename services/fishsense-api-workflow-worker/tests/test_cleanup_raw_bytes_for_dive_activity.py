# pylint: disable=unused-argument
"""Unit tests for cleanup_raw_bytes_for_dive_activity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    cleanup_raw_bytes_for_dive_activity as sut,
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


def _patch_routes(monkeypatch, delete_status: dict[str, int]):
    delete_calls: List[str] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            checksum = request.url.path.rsplit("/", 1)[-1].removesuffix(".ORF")
            delete_calls.append(checksum)
            return httpx.Response(delete_status.get(checksum, 204))
        return httpx.Response(405)

    transport = httpx.MockTransport(_mock_handler)
    real_async_client = httpx.AsyncClient

    def _client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(sut.httpx, "AsyncClient", _client_factory)
    return delete_calls


@pytest.mark.asyncio
async def test_deletes_all_raw_orfs_for_dive(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa"), _image(2, checksum="bbb")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    delete_calls = _patch_routes(monkeypatch, delete_status={})

    result = await ActivityEnvironment().run(
        sut.cleanup_raw_bytes_for_dive_activity, 42
    )

    assert result.deleted == 2
    assert set(delete_calls) == {"aaa", "bbb"}


@pytest.mark.asyncio
async def test_404_on_delete_counted_as_success(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    _patch_routes(monkeypatch, delete_status={"aaa": 404})

    result = await ActivityEnvironment().run(
        sut.cleanup_raw_bytes_for_dive_activity, 42
    )

    # nginx DAV returns 204 on delete-of-missing too, but 404 should
    # also count as a successful no-op so retried cleanups are
    # idempotent without raising.
    assert result.deleted == 1


@pytest.mark.asyncio
async def test_returns_zero_for_empty_dive(monkeypatch):
    fs = _make_fs([])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    _patch_routes(monkeypatch, delete_status={})

    result = await ActivityEnvironment().run(
        sut.cleanup_raw_bytes_for_dive_activity, 42
    )

    assert result.deleted == 0

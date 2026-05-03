# pylint: disable=unused-argument
"""Unit tests for stage_slate_pdf_activity."""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate import DiveSlate
from fishsense_api_workflow_worker.activities import stage_slate_pdf_activity as sut


def _slate(
    slate_id: int = 7, *, path: str = "/share/slate.pdf"
) -> DiveSlate:
    return DiveSlate(
        id=slate_id,
        name="test",
        dpi=300,
        path=path,
        created_at=None,
        reference_points=[(0.0, 0.0)],
    )


def _make_fs(slates: List[DiveSlate]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(return_value=slates)
    return fs


def _patch_routes(monkeypatch, head_present: bool):
    head_calls: List[int] = []
    put_calls: List[tuple] = []

    def _mock_handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "HEAD":
            head_calls.append(path)
            return httpx.Response(200 if head_present else 404)
        if request.method == "PUT":
            put_calls.append((path, request.content))
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
    real_path = sut.Path

    class _PathStub(real_path):  # type: ignore[misc]
        def read_bytes(self):  # type: ignore[override]
            return payload

    monkeypatch.setattr(sut, "Path", _PathStub)


@pytest.mark.asyncio
async def test_skips_nas_when_pdf_already_present(monkeypatch):
    fs = _make_fs([_slate(7)])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    nas = MagicMock()
    nas.download_to = MagicMock()
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    head_calls, put_calls = _patch_routes(monkeypatch, head_present=True)

    result = await ActivityEnvironment().run(
        sut.stage_slate_pdf_activity, 7
    )

    assert result is True
    nas.download_to.assert_not_called()
    assert not put_calls
    assert head_calls


@pytest.mark.asyncio
async def test_downloads_and_puts_when_pdf_missing(monkeypatch):
    fs = _make_fs([_slate(7)])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    nas = MagicMock()
    nas.download_to = MagicMock()
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"pdf-bytes")
    _, put_calls = _patch_routes(monkeypatch, head_present=False)

    result = await ActivityEnvironment().run(
        sut.stage_slate_pdf_activity, 7
    )

    assert result is True
    nas.download_to.assert_called_once()
    assert nas.download_to.call_args.kwargs["src_path"] == "/share/slate.pdf"
    assert len(put_calls) == 1


@pytest.mark.asyncio
async def test_raises_when_slate_missing(monkeypatch):
    fs = _make_fs([_slate(7)])  # only id=7 exists
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="not found"):
        await ActivityEnvironment().run(sut.stage_slate_pdf_activity, 99)


@pytest.mark.asyncio
async def test_raises_when_slate_path_is_empty(monkeypatch):
    fs = _make_fs([_slate(7, path="")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="no NAS path"):
        await ActivityEnvironment().run(sut.stage_slate_pdf_activity, 7)


@pytest.mark.asyncio
async def test_relative_slate_path_gets_prefixed_before_nas_download(monkeypatch):
    """Mirrors `stage_raw_bytes_for_dive_activity`: a share-relative
    `dive_slate.path` gets `e4e_nas.raw_root_path` prepended before
    reaching FileStation."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    fs = _make_fs([_slate(7, path="slates/v1/slate.pdf")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    nas = MagicMock()
    nas.download_to = MagicMock()
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"pdf-bytes")
    _patch_routes(monkeypatch, head_present=False)

    result = await ActivityEnvironment().run(sut.stage_slate_pdf_activity, 7)

    assert result is True
    nas.download_to.assert_called_once()
    assert nas.download_to.call_args.kwargs["src_path"] == (
        "/fishsense_data/REEF/data/slates/v1/slate.pdf"
    )

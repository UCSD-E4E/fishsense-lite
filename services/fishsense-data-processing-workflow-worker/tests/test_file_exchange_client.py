"""Unit tests for the worker-side FileExchangeClient that brokers raw
inputs and processed outputs through the nginx static_file_server.

These tests use httpx.MockTransport to assert URL shape, method, and
body roundtrip without standing up a real HTTP server.
"""

import httpx
import pytest

from fishsense_data_processing_workflow_worker.file_exchange import (
    FileExchangeClient,
)


_BASE = "http://static_file_server"


def _client(handler) -> FileExchangeClient:
    """Build a FileExchangeClient backed by a mocked httpx transport."""
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(base_url=_BASE, transport=transport)
    return FileExchangeClient(base_url=_BASE, http=http)


@pytest.mark.asyncio
async def test_download_raw_uses_exchange_raw_url_and_returns_bytes():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"\x00\x01\x02RAW")

    client = _client(handler)

    data = await client.download_raw(checksum="deadbeef")

    assert data == b"\x00\x01\x02RAW"
    assert len(seen) == 1
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/api/v1/exchange/raw/deadbeef.ORF"


@pytest.mark.asyncio
async def test_download_raw_raises_on_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = _client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.download_raw(checksum="missing")


@pytest.mark.asyncio
async def test_upload_processed_jpeg_puts_bytes_at_folder_checksum_url():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)

    client = _client(handler)

    await client.upload_processed_jpeg(
        folder="preprocess_groups_jpeg",
        checksum="cafef00d",
        data=b"JPEGBYTES",
    )

    assert len(seen) == 1
    req = seen[0]
    assert req.method == "PUT"
    assert req.url.path == "/api/v1/exchange/preprocess_groups_jpeg/cafef00d.JPG"
    assert req.content == b"JPEGBYTES"


@pytest.mark.asyncio
async def test_upload_processed_jpeg_raises_on_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.upload_processed_jpeg(
            folder="preprocess_groups_jpeg",
            checksum="cafef00d",
            data=b"JPEGBYTES",
        )


@pytest.mark.asyncio
async def test_upload_processed_jpeg_accepts_multiple_folders():
    """Stages 5.1 (headtail), 9 (slate), 0.1 (laser) all use the same
    PUT path with different folders. The folder is a string param."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)

    client = _client(handler)

    for folder in (
        "preprocess_groups_jpeg",
        "preprocess_headtail_jpeg",
        "preprocess_laser_jpeg",
        "preprocess_slate_images_jpeg",
    ):
        await client.upload_processed_jpeg(
            folder=folder, checksum="abc", data=b"x"
        )

    assert [r.url.path for r in seen] == [
        "/api/v1/exchange/preprocess_groups_jpeg/abc.JPG",
        "/api/v1/exchange/preprocess_headtail_jpeg/abc.JPG",
        "/api/v1/exchange/preprocess_laser_jpeg/abc.JPG",
        "/api/v1/exchange/preprocess_slate_images_jpeg/abc.JPG",
    ]

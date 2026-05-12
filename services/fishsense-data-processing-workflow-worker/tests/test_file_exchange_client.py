"""Unit tests for the worker-side FileExchangeClient that brokers raw
inputs and processed outputs through the nginx static_file_server.

These tests use httpx.MockTransport to assert URL shape, method, and
body roundtrip without standing up a real HTTP server.
"""

import httpx
import pytest

from fishsense_data_processing_workflow_worker.file_exchange import (
    FileExchangeClient,
    _basic_auth,
    open_file_exchange_client,
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
    def handler(_request: httpx.Request) -> httpx.Response:
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
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.upload_processed_jpeg(
            folder="preprocess_groups_jpeg",
            checksum="cafef00d",
            data=b"JPEGBYTES",
        )


@pytest.mark.asyncio
async def test_download_slate_pdf_uses_exchange_dive_slate_pdfs_url():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"%PDF-1.4 fake")

    client = _client(handler)

    data = await client.download_slate_pdf(slate_id=10)

    assert data == b"%PDF-1.4 fake"
    assert seen[0].method == "GET"
    assert seen[0].url.path == "/api/v1/exchange/dive_slate_pdfs/10.pdf"


@pytest.mark.asyncio
async def test_download_slate_pdf_raises_on_404():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = _client(handler)

    with pytest.raises(httpx.HTTPStatusError):
        await client.download_slate_pdf(slate_id=999)


@pytest.mark.asyncio
async def test_upload_processed_jpeg_accepts_multiple_folders():
    """Stages 5.1 (headtail), 9 (slate), 0.1 (laser) all use the same
    PUT path with different folders. The folder is a string param."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201)

    client = _client(handler)

    # The actual folders the data-worker writes (one per stage):
    # preprocess_jpeg (stage 0.1), preprocess_groups_jpeg (stage 2),
    # preprocess_headtail_jpeg (stage 5.1), preprocess_slate_images_jpeg
    # (stage 9). Test exercises URL construction across all four.
    for folder in (
        "preprocess_jpeg",
        "preprocess_groups_jpeg",
        "preprocess_headtail_jpeg",
        "preprocess_slate_images_jpeg",
    ):
        await client.upload_processed_jpeg(
            folder=folder, checksum="abc", data=b"x"
        )

    assert [r.url.path for r in seen] == [
        "/api/v1/exchange/preprocess_jpeg/abc.JPG",
        "/api/v1/exchange/preprocess_groups_jpeg/abc.JPG",
        "/api/v1/exchange/preprocess_headtail_jpeg/abc.JPG",
        "/api/v1/exchange/preprocess_slate_images_jpeg/abc.JPG",
    ]


# --- file-exchange auth + client wiring ---


class _FxSettings:
    """Stand-in for the Dynaconf ``settings.file_exchange`` section: a
    ``url`` attribute plus ``.get`` for the optional credential keys."""

    def __init__(self, url: str, **optional: str):
        self.url = url
        self._optional = optional

    def get(self, key, default=None):
        return self._optional.get(key, default)


def test_basic_auth_returns_none_when_no_creds():
    assert _basic_auth(None, None) is None


def test_basic_auth_requires_both_username_and_password():
    # A half-pair is treated as "no auth" rather than sending a broken header.
    assert _basic_auth("user", None) is None
    assert _basic_auth(None, "pass") is None
    assert _basic_auth("user", "") is None
    assert _basic_auth("", "pass") is None


def test_basic_auth_returns_basicauth_when_both_present():
    assert isinstance(_basic_auth("svc", "secret"), httpx.BasicAuth)


@pytest.mark.asyncio
async def test_open_file_exchange_client_without_creds_sends_no_auth():
    async with open_file_exchange_client(_FxSettings(_BASE)) as client:
        assert not isinstance(client._http.auth, httpx.BasicAuth)
        assert client._base_url == _BASE


@pytest.mark.asyncio
async def test_open_file_exchange_client_attaches_basic_auth_when_configured():
    settings = _FxSettings(
        "https://orchestrator.example", username="svc", password="secret"
    )
    async with open_file_exchange_client(settings) as client:
        assert isinstance(client._http.auth, httpx.BasicAuth)

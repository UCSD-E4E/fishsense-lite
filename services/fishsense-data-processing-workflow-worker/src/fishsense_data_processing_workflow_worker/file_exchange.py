"""Worker-side client for the nginx static_file_server file-exchange.

The exchange brokers files between the api-worker (NAS-facing) and the
data-worker (this service). URL contract:

    GET  /api/v1/exchange/raw/{checksum}.ORF             # raw input
    GET  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf # slate template
    PUT  /api/v1/exchange/{folder}/{checksum}.JPG        # processed output

Where `{folder}` is one of `preprocess_jpeg` (stage 0.1 / laser),
`preprocess_groups_jpeg` (stage 2 / dive species),
`preprocess_headtail_jpeg` (stage 5.1), or
`preprocess_slate_images_jpeg` (stage 9). The labeler-facing GET
routes in `deploy/static_file_server/nginx.conf` rewrite these to
their virtual-folder names (`preprocess_jpeg`, `groups_jpeg`,
`headtail_jpeg`, `dive_slate_jpgs`) for the LS task URLs.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

# Per-request timeout for the file-exchange. Raw ORFs are ~20 MB and
# processed JPEGs a few MB; 60 s comfortably covers a transfer plus the
# Traefik/authentik round-trip in production.
_EXCHANGE_TIMEOUT = httpx.Timeout(60.0)


def _basic_auth(
    username: str | None, password: str | None
) -> httpx.BasicAuth | None:
    """HTTP Basic auth for the exchange, or None when creds aren't set.

    Both are required — half a credential pair is treated as "no auth"
    so a misconfigured-but-non-empty value can't silently send a broken
    header. The local devcontainer leaves both unset and hits nginx
    directly; production sets both for the authentik-fronted route.
    """
    if username and password:
        return httpx.BasicAuth(username, password)
    return None


class FileExchangeClient:
    """Thin async wrapper around an httpx.AsyncClient for the worker
    file-exchange. Constructed per-activity-call; not a singleton."""

    def __init__(self, base_url: str, http: httpx.AsyncClient):
        self._base_url = base_url.rstrip("/")
        self._http = http

    async def download_raw(self, checksum: str) -> bytes:
        response = await self._http.get(f"/api/v1/exchange/raw/{checksum}.ORF")
        response.raise_for_status()
        return response.content

    async def download_slate_pdf(self, slate_id: int) -> bytes:
        response = await self._http.get(
            f"/api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf"
        )
        response.raise_for_status()
        return response.content

    async def upload_processed_jpeg(
        self, folder: str, checksum: str, data: bytes
    ) -> None:
        response = await self._http.put(
            f"/api/v1/exchange/{folder}/{checksum}.JPG",
            content=data,
        )
        response.raise_for_status()


@asynccontextmanager
async def open_file_exchange_client(
    file_exchange_settings,
) -> AsyncIterator[FileExchangeClient]:
    """Yield a FileExchangeClient wired to the configured exchange URL,
    timeout, and (optional) HTTP Basic auth, managing the httpx client
    lifecycle for the duration of the ``async with`` block.

    ``file_exchange_settings`` is the ``settings.file_exchange`` section:
    a ``url`` attribute plus optional ``username`` / ``password`` keys
    (read via ``.get`` so an unset pair just means no auth).
    """
    url = file_exchange_settings.url
    auth = _basic_auth(
        file_exchange_settings.get("username"),
        file_exchange_settings.get("password"),
    )
    async with httpx.AsyncClient(
        base_url=url, timeout=_EXCHANGE_TIMEOUT, auth=auth
    ) as http:
        yield FileExchangeClient(base_url=url, http=http)

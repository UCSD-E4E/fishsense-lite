"""Worker-side client for the nginx static_file_server file-exchange.

The exchange brokers files between the api-worker (NAS-facing) and the
data-worker (this service). URL contract:

    GET  /api/v1/exchange/raw/{checksum}.ORF             # raw input
    GET  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf # slate template
    PUT  /api/v1/exchange/{folder}/{checksum}.JPG        # processed output

Where `{folder}` is one of `preprocess_groups_jpeg`,
`preprocess_headtail_jpeg`, `preprocess_laser_jpeg`,
`preprocess_slate_images_jpeg` — matching the existing labeler-facing
GET routes in deploy/static_file_server/nginx.conf.
"""

import httpx


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

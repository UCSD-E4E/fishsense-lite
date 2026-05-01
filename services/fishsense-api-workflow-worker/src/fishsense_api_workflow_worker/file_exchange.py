"""api-worker side of the nginx static_file_server file-exchange.

Mirrors the data-worker's `FileExchangeClient` shape (see
`fishsense_data_processing_workflow_worker/file_exchange.py`).
Covers both the Phase 3a staging-in path (HEAD + PUT raw `.ORF`
and slate PDF) and the Phase 3b archive/cleanup path (GET processed
JPEG, DELETE raw `.ORF`).

URL contract this worker uses:

    HEAD   /api/v1/exchange/raw/{checksum}.ORF
    PUT    /api/v1/exchange/raw/{checksum}.ORF
    DELETE /api/v1/exchange/raw/{checksum}.ORF
    HEAD   /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf
    PUT    /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf
    GET    /api/v1/exchange/{folder}/{checksum}.JPG
"""

from __future__ import annotations

import httpx


class StagingFileExchangeClient:
    """Async wrapper for the file-exchange endpoints the api-worker
    needs. Constructed per-activity-call; not a singleton."""

    def __init__(self, base_url: str, http: httpx.AsyncClient):
        self._base_url = base_url.rstrip("/")
        self._http = http

    # ----- Phase 3a: staging in -----

    async def has_raw(self, checksum: str) -> bool:
        """True if `/api/v1/exchange/raw/{checksum}.ORF` already exists.
        Lets the staging activity skip already-staged checksums on
        retried/idempotent runs without re-downloading from NAS."""
        response = await self._http.head(
            f"/api/v1/exchange/raw/{checksum}.ORF"
        )
        return response.status_code == 200

    async def upload_raw(self, checksum: str, data: bytes) -> None:
        response = await self._http.put(
            f"/api/v1/exchange/raw/{checksum}.ORF",
            content=data,
        )
        response.raise_for_status()

    async def has_slate_pdf(self, slate_id: int) -> bool:
        response = await self._http.head(
            f"/api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf"
        )
        return response.status_code == 200

    async def upload_slate_pdf(self, slate_id: int, data: bytes) -> None:
        response = await self._http.put(
            f"/api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf",
            content=data,
        )
        response.raise_for_status()

    # ----- Phase 3b: archive + cleanup -----

    async def download_processed_jpeg(
        self, folder: str, checksum: str
    ) -> bytes:
        response = await self._http.get(
            f"/api/v1/exchange/{folder}/{checksum}.JPG"
        )
        response.raise_for_status()
        return response.content

    async def delete_raw(self, checksum: str) -> bool:
        """DELETE the raw `.ORF` entry. Returns True if the file is
        gone after this call (whether deleted or already absent),
        False on a hard error so the caller can decide."""
        response = await self._http.delete(
            f"/api/v1/exchange/raw/{checksum}.ORF"
        )
        if response.status_code in (200, 204, 404):
            return True
        response.raise_for_status()
        return False

"""api-worker side of the nginx static_file_server file-exchange.

Mirrors the data-worker's `FileExchangeClient` shape (see
`fishsense_data_processing_workflow_worker/file_exchange.py`) but
for the staging-in path: HEAD-check + PUT for raw `.ORF` and slate
PDFs.

URL contract this worker uses:

    HEAD /api/v1/exchange/raw/{checksum}.ORF              # idempotency check
    PUT  /api/v1/exchange/raw/{checksum}.ORF              # stage from NAS
    HEAD /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf  # idempotency check
    PUT  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf  # stage from NAS

Phase 3b will add `download_processed_jpeg` + `delete_raw` for the
archive / cleanup side.
"""

from __future__ import annotations

import httpx


class StagingFileExchangeClient:
    """Async wrapper for the staging-side endpoints. Constructed
    per-activity-call; not a singleton."""

    def __init__(self, base_url: str, http: httpx.AsyncClient):
        self._base_url = base_url.rstrip("/")
        self._http = http

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

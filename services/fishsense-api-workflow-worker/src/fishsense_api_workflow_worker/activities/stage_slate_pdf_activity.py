"""Activity to stage a slate template PDF from NAS to the file-exchange.

Stage 9 (slate preprocess) reads the slate PDF off
`/api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf` to overlay it on
each rectified frame. This activity hydrates that endpoint from the
NAS path stored in `DiveSlate.path`.

Idempotent: HEAD-checks the file-exchange first and skips if the
PDF is already present.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.file_exchange import StagingFileExchangeClient
from fishsense_api_workflow_worker.nas import NasDownloadClient


def _build_nas_client() -> NasDownloadClient:
    return NasDownloadClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


@activity.defn
async def stage_slate_pdf_activity(slate_id: int) -> bool:
    """Stage `slate_id`'s PDF from NAS. Returns True if the PDF is
    on the file-exchange after this call (whether already-present or
    newly staged); raises if the slate row / NAS path is missing or
    the upload fails."""
    async with get_fs_client() as fs:
        slates = await fs.dive_slates.get() or []
        slate = next((s for s in slates if s.id == slate_id), None)
        if slate is None:
            raise ValueError(f"slate_id={slate_id} not found")
        if not slate.path:
            raise ValueError(f"slate_id={slate_id} has no NAS path")

    async with httpx.AsyncClient(
        base_url=settings.file_exchange.url,
        timeout=httpx.Timeout(60.0),
    ) as http:
        exchange = StagingFileExchangeClient(
            base_url=settings.file_exchange.url, http=http
        )

        if await exchange.has_slate_pdf(slate_id):
            activity.logger.info(
                "slate_id=%d already staged on file-exchange; skipping NAS fetch",
                slate_id,
            )
            return True

        nas = _build_nas_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            await asyncio.to_thread(
                nas.download_to,
                src_path=slate.path,
                dest_dir=tmpdir,
            )
            local_path = Path(tmpdir) / os.path.basename(slate.path)
            data = await asyncio.to_thread(local_path.read_bytes)
            await exchange.upload_slate_pdf(slate_id, data)

    activity.logger.info("staged slate_id=%d to file-exchange", slate_id)
    return True

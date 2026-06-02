"""Activity to stage a slate template PDF from NAS to the Garage store.

Stage 9 (slate preprocess) reads the slate PDF off the Garage
`slate_pdf/{slate_id}.pdf` scratch key to overlay it on each rectified
frame. This activity hydrates that key from the NAS path stored in
`DiveSlate.path`.

Path resolution: same shape as `stage_raw_bytes_for_dive_activity` —
`DiveSlate.path` in the DB is share-relative; this activity prepends
`e4e_nas.raw_root_path` before calling FileStation.

NAS access is read-only (download). Idempotent: HEAD-checks the object
store first (HeadObject) and skips if the PDF is already present.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.object_store import open_object_store_client
from fishsense_api_workflow_worker.nas import NasDownloadClient


def _build_nas_client() -> NasDownloadClient:
    return NasDownloadClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


def _resolve_nas_path(relative_path: str) -> str:
    """Prepend `e4e_nas.raw_root_path` to a share-relative DB path.

    Mirrors the helper in `stage_raw_bytes_for_dive_activity` —
    duplicated rather than shared because both activities are tiny and
    a shared util would be the only consumer. Refactor when a third
    consumer appears.
    """
    if relative_path.startswith("/"):
        return relative_path
    root = settings.e4e_nas.raw_root_path.rstrip("/")
    return f"{root}/{relative_path.lstrip('/')}"


@activity.defn
async def stage_slate_pdf_activity(slate_id: int) -> bool:
    """Stage `slate_id`'s PDF from NAS. Returns True if the PDF is in
    the object store after this call (whether already-present or newly
    staged); raises if the slate row / NAS path is missing or the
    upload fails."""
    async with get_fs_client() as fs:
        slates = await fs.dive_slates.get() or []
        slate = next((s for s in slates if s.id == slate_id), None)
        if slate is None:
            raise ValueError(f"slate_id={slate_id} not found")
        if not slate.path:
            raise ValueError(f"slate_id={slate_id} has no NAS path")

    exchange = open_object_store_client()

    if await exchange.has_slate_pdf(slate_id):
        activity.logger.info(
            "slate_id=%d already staged in object store; skipping NAS fetch",
            slate_id,
        )
        return True

    nas = _build_nas_client()
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = _resolve_nas_path(slate.path)
        await asyncio.to_thread(
            nas.download_to,
            src_path=src_path,
            dest_dir=tmpdir,
        )
        local_path = Path(tmpdir) / os.path.basename(src_path)
        data = await asyncio.to_thread(local_path.read_bytes)
        await exchange.upload_slate_pdf(slate_id, data)

    activity.logger.info("staged slate_id=%d to object store", slate_id)
    return True

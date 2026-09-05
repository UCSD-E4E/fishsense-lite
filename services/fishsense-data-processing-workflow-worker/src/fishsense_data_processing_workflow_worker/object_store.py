"""data-worker side of the Garage (S3-compatible) object store.

The key contract and the S3 primitives live in
`fishsense_shared.object_store` — it is an agreement *between* this
worker and the api-worker, so neither owns it. This module is only the
data-worker's method subset layered on top.

This worker reads staged raw inputs + slate PDFs from the scratch bucket
and writes processed JPEGs to the labels bucket. It has **no NAS access**
by design, and no way to delete anything.

The ``{jpeg_prefix}`` values are the same folder names the workflows
already pass as ``output_folder`` — ``preprocess_jpeg`` (0.1),
``preprocess_groups_jpeg`` (2), ``preprocess_headtail_jpeg`` (5.1),
``preprocess_slate_images_jpeg`` (9).
"""

from __future__ import annotations

from fishsense_shared.object_store import (
    RAW_PREFIX,
    SLATE_PDF_PREFIX,
    BaseObjectStoreClient,
    jpeg_key,
    open_client,
    raw_key,
    slate_pdf_key,
)

__all__ = [
    "RAW_PREFIX",
    "SLATE_PDF_PREFIX",
    "ObjectStoreClient",
    "jpeg_key",
    "open_client",
    "open_object_store_client",
    "raw_key",
    "slate_pdf_key",
]


def open_object_store_client() -> "ObjectStoreClient":
    """Build an ``ObjectStoreClient`` from worker settings.

    The ``config`` import is function-local so importing this module
    doesn't eagerly trigger Dynaconf validation — only calling this at
    activity runtime does. The 4 preprocess activities call this.
    """
    from fishsense_data_processing_workflow_worker.config import settings

    return open_client(settings, ObjectStoreClient)


class ObjectStoreClient(BaseObjectStoreClient):
    """The data-worker's read + JPEG-write vocabulary.

    Reads raw/slate **scratch** from ``bucket``; writes processed JPEGs to
    ``labels_bucket`` (the LS-facing bucket) under ``labels_prefix``.
    ``labels_bucket`` defaults to ``bucket`` so single-bucket layouts keep
    working unchanged.
    """

    async def download_raw(self, checksum: str) -> bytes:
        return await self._get(raw_key(checksum))

    async def download_slate_pdf(self, slate_id: int) -> bytes:
        return await self._get(slate_pdf_key(slate_id))

    async def upload_processed_jpeg(
        self, folder: str, checksum: str, data: bytes
    ) -> None:
        await self._put(
            jpeg_key(folder, checksum, self._labels_prefix),
            data,
            bucket=self._labels_bucket,
        )

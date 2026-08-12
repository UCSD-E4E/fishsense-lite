"""api-worker side of the Garage (S3-compatible) object store.

The key contract and the S3 primitives live in
`fishsense_shared.object_store` — it is an agreement *between* this
worker and the data-worker, so neither owns it. This module is only the
api-worker's method subset layered on top.

This worker stages raw `.ORF` + slate PDFs *in* (HEAD + PUT) and cleans
up the ``raw/`` scratch prefix afterwards (DELETE). It never writes
JPEGs — it only reads their presence, to gate the decoupled species
populate.

NAS safety invariant: the only deletes this client can issue target the
Garage scratch prefixes. The NAS stays the read-only source of truth —
nothing here (or anywhere on the api-worker) deletes from the NAS.
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
    doesn't eagerly trigger Dynaconf validation (see the config
    gotcha in CLAUDE.md) — only calling this at activity runtime does.
    Activities call this; tests patch it to inject a moto-backed client.
    """
    # pylint: disable=import-outside-toplevel
    from fishsense_api_workflow_worker.config import settings

    return open_client(settings, ObjectStoreClient)


class ObjectStoreClient(BaseObjectStoreClient):
    """The api-worker's staging + scratch-cleanup vocabulary.

    Staging (raw/slate) lives in ``bucket`` (scratch); processed JPEGs that
    Label Studio serves live in ``labels_bucket`` under ``labels_prefix``.
    ``labels_bucket`` defaults to ``bucket`` so single-bucket layouts keep
    working unchanged.
    """

    # ----- staging in -----

    async def has_raw(self, checksum: str) -> bool:
        return await self._exists(raw_key(checksum))

    async def has_processed_jpeg(self, folder: str, checksum: str) -> bool:
        """True iff the data-worker has already written this stage's
        processed JPEG to Garage. Used by the scheduled species-populate
        activity to gate task import on the JPEG existing (a decoupled
        populate must never seed rows for an image whose JPEG isn't
        written yet — that would drop the dive out of the preprocess
        cohort with a broken image). Checks the **labels** bucket/prefix
        where the data-worker writes JPEGs."""
        return await self._exists(
            jpeg_key(folder, checksum, self._labels_prefix),
            bucket=self._labels_bucket,
        )

    async def upload_raw(self, checksum: str, data: bytes) -> None:
        await self._put(raw_key(checksum), data)

    async def has_slate_pdf(self, slate_id: int) -> bool:
        return await self._exists(slate_pdf_key(slate_id))

    async def upload_slate_pdf(self, slate_id: int, data: bytes) -> None:
        await self._put(slate_pdf_key(slate_id), data)

    async def download_slate_pdf(self, slate_id: int) -> bytes:
        """Read a staged slate template PDF back from Garage.

        The dive-slate sync (panel-offset strip) and the slate populate
        (pre-annotation composite conversion) both need the template's
        aspect ratio; both read it through here.
        """
        return await self._get(slate_pdf_key(slate_id))

    # ----- scratch cleanup (Garage only — NEVER the NAS) -----

    async def delete_raw(self, checksum: str) -> bool:
        """Delete the staged raw `.ORF` *scratch* object from Garage.

        Returns True (delete is idempotent). This only ever removes the
        Garage scratch copy — the NAS source `.ORF` is never touched.
        """
        await self._delete(raw_key(checksum))
        return True

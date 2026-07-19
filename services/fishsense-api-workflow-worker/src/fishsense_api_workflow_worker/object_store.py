"""api-worker side of the Garage (S3-compatible) object store.

Replaces the nginx ``static_file_server`` file-exchange. One bucket,
content-type prefixes — the cross-worker key contract:

    raw/{checksum}.ORF            # staged raw input (scratch)
    slate_pdf/{slate_id}.pdf      # staged slate template (scratch)
    {jpeg_prefix}/{checksum}.JPG  # processed output (durable), written
                                  # by the data-worker; LS reads via presign

This worker stages raw `.ORF` + slate PDFs *in* (HEAD + PUT) and cleans
up the ``raw/`` scratch prefix afterwards (DELETE).

NAS safety invariant: the only deletes this client can issue target the
Garage scratch prefixes. The NAS stays the read-only source of truth —
nothing here (or anywhere on the api-worker) deletes from the NAS.

Mirrors the data-worker's ``ObjectStoreClient``
(``fishsense_data_processing_workflow_worker/object_store.py``); the key
helpers are duplicated, like the two ``file_exchange.py`` clients were,
because each worker uses a different method subset.
"""

from __future__ import annotations

import asyncio

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

RAW_PREFIX = "raw"
SLATE_PDF_PREFIX = "slate_pdf"

# botocore surfaces a missing key as one of these `Error.Code` values
# depending on whether the call was HeadObject (404/NotFound) or
# GetObject (NoSuchKey).
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}

__all__ = [
    "RAW_PREFIX",
    "SLATE_PDF_PREFIX",
    "ObjectStoreClient",
    "build_s3_client",
    "open_object_store_client",
    "raw_key",
    "slate_pdf_key",
]


def raw_key(checksum: str) -> str:
    return f"{RAW_PREFIX}/{checksum}.ORF"


def slate_pdf_key(slate_id: int) -> str:
    return f"{SLATE_PDF_PREFIX}/{slate_id}.pdf"


def processed_jpeg_key(folder: str, checksum: str, prefix: str = "") -> str:
    """Physical Garage key for a data-worker-written processed JPEG.

    `folder` is the per-stage prefix (`preprocess_groups_jpeg`,
    `preprocess_jpeg`, `preprocess_headtail_jpeg`,
    `preprocess_slate_images_jpeg`). `prefix` is the optional
    `labels_prefix` that partitions our JPEGs within the shared labels
    bucket (mirrors the coral-gardeners prefix); empty → no prefix.
    """
    base = f"{folder}/{checksum}.JPG"
    prefix = (prefix or "").strip("/")
    return f"{prefix}/{base}" if prefix else base


def build_s3_client(
    *, endpoint_url: str, region: str, access_key: str, secret_key: str
):
    """Build a boto3 S3 client pointed at Garage.

    Garage requires **path-style** addressing (it has no virtual-host
    bucket DNS) and an explicit endpoint + region; SigV4 is its default
    signature scheme.
    """
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def open_object_store_client() -> "ObjectStoreClient":
    """Build an ``ObjectStoreClient`` from worker settings.

    The ``config`` import is function-local so importing this module
    doesn't eagerly trigger Dynaconf validation (see the config
    gotcha in CLAUDE.md) — only calling this at activity runtime does.
    Activities call this; tests patch it to inject a moto-backed client.
    """
    # pylint: disable=import-outside-toplevel
    from fishsense_api_workflow_worker.config import settings

    s3 = build_s3_client(
        endpoint_url=settings.object_store.endpoint_url,
        region=settings.object_store.region,
        access_key=settings.object_store.access_key,
        secret_key=settings.object_store.secret_key,
    )
    return ObjectStoreClient(
        s3,
        settings.object_store.bucket,
        labels_bucket=settings.object_store.get("labels_bucket", None),
        labels_prefix=settings.object_store.get("labels_prefix", "") or "",
    )


class ObjectStoreClient:
    """Thin async wrapper over a boto3 S3 client for the api-worker's
    staging + scratch-cleanup needs. Constructed per-activity-call; the
    boto3 client is injected so tests can pass a moto-backed one.

    Staging (raw/slate) lives in ``bucket`` (scratch); processed JPEGs that
    Label Studio serves live in ``labels_bucket`` under ``labels_prefix``.
    ``labels_bucket`` defaults to ``bucket`` so single-bucket layouts keep
    working unchanged."""

    def __init__(self, s3, bucket: str, labels_bucket=None, labels_prefix=""):
        self._s3 = s3
        self._bucket = bucket
        self._labels_bucket = labels_bucket or bucket
        self._labels_prefix = labels_prefix or ""

    async def _exists(self, key: str, bucket: str | None = None) -> bool:
        target = bucket or self._bucket

        def _do() -> bool:
            try:
                self._s3.head_object(Bucket=target, Key=key)
                return True
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in _NOT_FOUND_CODES:
                    return False
                raise

        return await asyncio.to_thread(_do)

    async def _put(self, key: str, data: bytes) -> None:
        await asyncio.to_thread(
            self._s3.put_object, Bucket=self._bucket, Key=key, Body=data
        )

    async def _delete(self, key: str) -> None:
        # S3 delete_object is idempotent: deleting an absent key returns
        # success (no ClientError), so retries are naturally safe.
        await asyncio.to_thread(
            self._s3.delete_object, Bucket=self._bucket, Key=key
        )

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
            processed_jpeg_key(folder, checksum, self._labels_prefix),
            bucket=self._labels_bucket,
        )

    async def upload_raw(self, checksum: str, data: bytes) -> None:
        await self._put(raw_key(checksum), data)

    async def has_slate_pdf(self, slate_id: int) -> bool:
        return await self._exists(slate_pdf_key(slate_id))

    async def upload_slate_pdf(self, slate_id: int, data: bytes) -> None:
        await self._put(slate_pdf_key(slate_id), data)

    # ----- scratch cleanup (Garage only — NEVER the NAS) -----

    async def delete_raw(self, checksum: str) -> bool:
        """Delete the staged raw `.ORF` *scratch* object from Garage.

        Returns True (delete is idempotent). This only ever removes the
        Garage scratch copy — the NAS source `.ORF` is never touched.
        """
        await self._delete(raw_key(checksum))
        return True

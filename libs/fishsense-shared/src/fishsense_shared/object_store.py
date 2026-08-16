"""Garage (S3-compatible) object-store contract shared by both workers.

This is the **single source of truth** for the cross-worker key contract.
It lives here, next to `preprocess_contracts`, for the same reason those
DTOs do: it is an agreement *between* the api-worker and the data-worker,
so neither one gets to own it.

    raw/{checksum}.ORF            # scratch; api-worker PUTs, data-worker GETs
    slate_pdf/{slate_id}.pdf      # scratch; api-worker PUTs, data-worker GETs
    {prefix}/{folder}/{checksum}.JPG   # durable; data-worker PUTs, LS presigns

`{folder}` is the per-stage JPEG prefix — `preprocess_jpeg` (0.1),
`preprocess_groups_jpeg` (2), `preprocess_headtail_jpeg` (5.1),
`preprocess_slate_images_jpeg` (9). `{prefix}` is the optional
`labels_prefix` partitioning our objects inside a shared labels bucket.

Each worker subclasses `BaseObjectStoreClient` and exposes only the
method subset it is allowed to use: the api-worker stages scratch in and
deletes it afterwards; the data-worker reads scratch and writes JPEGs.
That asymmetry is a real safety boundary, which is why the base class
holds the primitives and the subclasses hold the vocabulary.

**NAS safety invariant**: nothing in this module can touch the NAS. The
only deletes reachable from here target Garage scratch keys.

boto3 is an optional extra of `fishsense-shared` (`[s3]`) — importing this
module requires it, but importing the package does not.
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
NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})

__all__ = [
    "NOT_FOUND_CODES",
    "RAW_PREFIX",
    "SLATE_PDF_PREFIX",
    "BaseObjectStoreClient",
    "build_s3_client",
    "jpeg_key",
    "open_client",
    "raw_key",
    "slate_pdf_key",
]


def raw_key(checksum: str) -> str:
    """Physical Garage key for a staged raw `.ORF`."""
    return f"{RAW_PREFIX}/{checksum}.ORF"


def slate_pdf_key(slate_id: int) -> str:
    """Physical Garage key for a staged slate-template PDF."""
    return f"{SLATE_PDF_PREFIX}/{slate_id}.pdf"


def jpeg_key(folder: str, checksum: str, prefix: str = "") -> str:
    """Physical Garage key for a data-worker-written processed JPEG.

    `folder` is the per-stage prefix; `prefix` is the optional
    `labels_prefix` that partitions our JPEGs within a shared labels
    bucket. Surrounding slashes are stripped so a caller can't produce a
    double-slash key, which S3 treats as a different object.
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


def open_client(settings, client_cls):
    """Build `client_cls` from a worker's Dynaconf `settings`.

    `client_cls` is the caller's `BaseObjectStoreClient` subclass — the
    api-worker's staging vocabulary or the data-worker's read/write one.
    Both read the identical `[object_store]` section, so the mapping from
    settings to constructor lives here rather than twice.

    `labels_bucket` / `labels_prefix` are optional: a single-bucket
    deployment sets neither, and `BaseObjectStoreClient` falls back to
    `bucket` / `""`. `or ""` guards the TOML-empty-string-as-None path,
    which would otherwise put the literal "None" in every JPEG key.

    Callers keep their own thin `open_object_store_client()` wrapper so
    the `config` import stays function-local — importing a worker's
    `object_store` must not eagerly trigger Dynaconf validation (see the
    config gotcha in CLAUDE.md).
    """
    s3 = build_s3_client(
        endpoint_url=settings.object_store.endpoint_url,
        region=settings.object_store.region,
        access_key=settings.object_store.access_key,
        secret_key=settings.object_store.secret_key,
    )
    return client_cls(
        s3,
        settings.object_store.bucket,
        labels_bucket=settings.object_store.get("labels_bucket", None),
        labels_prefix=settings.object_store.get("labels_prefix", "") or "",
    )


class BaseObjectStoreClient:
    """Async primitives over an injected boto3 S3 client.

    Constructed per-activity-call; the boto3 client is injected so tests
    can pass a moto-backed one. Every call is bounced through
    `asyncio.to_thread` because boto3 is synchronous and these run inside
    Temporal activities on the event loop.

    Scratch (raw/slate) lives in `bucket`; the processed JPEGs Label
    Studio serves live in `labels_bucket` under `labels_prefix`.
    `labels_bucket` defaults to `bucket`, so single-bucket layouts keep
    working unchanged.
    """

    def __init__(self, s3, bucket: str, labels_bucket=None, labels_prefix=""):
        self._s3 = s3
        self._bucket = bucket
        self._labels_bucket = labels_bucket or bucket
        self._labels_prefix = labels_prefix or ""

    async def _exists(self, key: str, bucket: str | None = None) -> bool:
        """HeadObject, mapping only *not-found* to False.

        Any other error (403, 500, …) propagates: reporting them as
        "absent" would make staging re-upload forever and make cleanup
        believe there was nothing to delete.
        """
        target = bucket or self._bucket

        def _do() -> bool:
            try:
                self._s3.head_object(Bucket=target, Key=key)
                return True
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code in NOT_FOUND_CODES:
                    return False
                raise

        return await asyncio.to_thread(_do)

    async def _get(self, key: str, bucket: str | None = None) -> bytes:
        target = bucket or self._bucket

        def _do() -> bytes:
            response = self._s3.get_object(Bucket=target, Key=key)
            # Close the StreamingBody so botocore returns the underlying
            # HTTP connection to its pool; leaking it across repeated
            # downloads can exhaust the pool and stall activities.
            body = response["Body"]
            try:
                return body.read()
            finally:
                body.close()

        return await asyncio.to_thread(_do)

    async def _put(self, key: str, data: bytes, bucket: str | None = None) -> None:
        target = bucket or self._bucket
        await asyncio.to_thread(
            self._s3.put_object, Bucket=target, Key=key, Body=data
        )

    async def _delete(self, key: str, bucket: str | None = None) -> None:
        # S3 delete_object is idempotent: deleting an absent key returns
        # success (no ClientError), so retries are naturally safe.
        target = bucket or self._bucket
        await asyncio.to_thread(
            self._s3.delete_object, Bucket=target, Key=key
        )

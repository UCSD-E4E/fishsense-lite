"""data-worker side of the Garage (S3-compatible) object store.

Mirror of the api-worker's ``ObjectStoreClient``
(``fishsense_api_workflow_worker/object_store.py``). The data-worker
reads staged raw inputs + slate PDFs and writes processed JPEGs; it has
**no NAS access** by design.

Key contract (shared with the api-worker):

    raw/{checksum}.ORF            # read
    slate_pdf/{slate_id}.pdf      # read (stage 9)
    {jpeg_prefix}/{checksum}.JPG  # write (durable; LS reads via presign)

The ``{jpeg_prefix}`` values are the same folder names the workflows
already pass as ``output_folder`` — ``preprocess_jpeg`` (0.1),
``preprocess_groups_jpeg`` (2), ``preprocess_headtail_jpeg`` (5.1),
``preprocess_slate_images_jpeg`` (9).
"""

from __future__ import annotations

import asyncio

import boto3
from botocore.config import Config

RAW_PREFIX = "raw"
SLATE_PDF_PREFIX = "slate_pdf"

__all__ = [
    "RAW_PREFIX",
    "SLATE_PDF_PREFIX",
    "ObjectStoreClient",
    "build_s3_client",
    "jpeg_key",
    "open_object_store_client",
    "raw_key",
    "slate_pdf_key",
]


def raw_key(checksum: str) -> str:
    return f"{RAW_PREFIX}/{checksum}.ORF"


def slate_pdf_key(slate_id: int) -> str:
    return f"{SLATE_PDF_PREFIX}/{slate_id}.pdf"


def jpeg_key(folder: str, checksum: str) -> str:
    return f"{folder}/{checksum}.JPG"


def build_s3_client(
    *, endpoint_url: str, region: str, access_key: str, secret_key: str
):
    """Build a boto3 S3 client pointed at Garage.

    Garage requires **path-style** addressing (no virtual-host bucket
    DNS) and an explicit endpoint + region; SigV4 is its default.
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
    doesn't eagerly trigger Dynaconf validation — only calling this at
    activity runtime does. The 4 preprocess activities call this.
    """
    # pylint: disable=import-outside-toplevel
    from fishsense_data_processing_workflow_worker.config import settings

    s3 = build_s3_client(
        endpoint_url=settings.object_store.endpoint_url,
        region=settings.object_store.region,
        access_key=settings.object_store.access_key,
        secret_key=settings.object_store.secret_key,
    )
    return ObjectStoreClient(s3, settings.object_store.bucket)


class ObjectStoreClient:
    """Thin async wrapper over a boto3 S3 client for the data-worker's
    read + JPEG-write needs. Constructed per-activity-call; the boto3
    client is injected so tests can pass a moto-backed one."""

    def __init__(self, s3, bucket: str):
        self._s3 = s3
        self._bucket = bucket

    async def _get(self, key: str) -> bytes:
        def _do() -> bytes:
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read()

        return await asyncio.to_thread(_do)

    async def download_raw(self, checksum: str) -> bytes:
        return await self._get(raw_key(checksum))

    async def download_slate_pdf(self, slate_id: int) -> bytes:
        return await self._get(slate_pdf_key(slate_id))

    async def upload_processed_jpeg(
        self, folder: str, checksum: str, data: bytes
    ) -> None:
        await asyncio.to_thread(
            self._s3.put_object,
            Bucket=self._bucket,
            Key=jpeg_key(folder, checksum),
            Body=data,
        )

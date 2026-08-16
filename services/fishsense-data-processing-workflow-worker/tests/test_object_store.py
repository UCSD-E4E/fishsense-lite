"""Unit tests for the data-worker ObjectStoreClient (moto-backed).

Mirrors test_file_exchange_client.py but for S3/Garage. Pins:
  1. download_raw reads raw/{checksum}.ORF and returns the bytes.
  2. download_slate_pdf reads slate_pdf/{slate_id}.pdf.
  3. upload_processed_jpeg writes {folder}/{checksum}.JPG for each of
     the four stage output folders.
  4. The key contract matches the api-worker's writer side.
"""

from __future__ import annotations

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from temporalio.testing import ActivityEnvironment

from fishsense_data_processing_workflow_worker import object_store as sut

BUCKET = "fishsense-test"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


# The key contract (`raw_key` / `slate_pdf_key` / `jpeg_key`) and
# `build_s3_client`'s path-style addressing now live in
# `fishsense_shared.object_store` and are pinned by
# `libs/fishsense-shared/tests/test_object_store.py`. Asserting them again
# here would be the same duplication this module was just refactored out of.
# What stays below is what is genuinely this worker's: its method subset.


@pytest.mark.asyncio
async def test_download_raw_returns_bytes(s3):
    s3.put_object(Bucket=BUCKET, Key="raw/deadbeef.ORF", Body=b"\x00\x01RAW")
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        return await client.download_raw("deadbeef")

    assert await ActivityEnvironment().run(_run) == b"\x00\x01RAW"


@pytest.mark.asyncio
async def test_download_slate_pdf_returns_bytes(s3):
    s3.put_object(Bucket=BUCKET, Key="slate_pdf/5.pdf", Body=b"%PDF-1.7")
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        return await client.download_slate_pdf(5)

    assert await ActivityEnvironment().run(_run) == b"%PDF-1.7"


@pytest.mark.parametrize(
    "folder",
    [
        "preprocess_jpeg",
        "preprocess_groups_jpeg",
        "preprocess_headtail_jpeg",
        "preprocess_slate_images_jpeg",
    ],
)
@pytest.mark.asyncio
async def test_upload_processed_jpeg_writes_folder_checksum_key(s3, folder):
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        await client.upload_processed_jpeg(
            folder=folder, checksum="cafef00d", data=b"JPEGBYTES"
        )

    await ActivityEnvironment().run(_run)

    key = f"{folder}/cafef00d.JPG"
    assert s3.get_object(Bucket=BUCKET, Key=key)["Body"].read() == b"JPEGBYTES"


@pytest.mark.asyncio
async def test_split_buckets_read_scratch_write_labels(s3):
    # Two-bucket layout: raw scratch is read from `bucket`, JPEGs are written
    # to `labels_bucket` under `labels_prefix`.
    labels = "labels-fishsense-test"
    s3.create_bucket(Bucket=labels)
    s3.put_object(Bucket=BUCKET, Key="raw/deadbeef.ORF", Body=b"RAW")
    client = sut.ObjectStoreClient(
        s3, BUCKET, labels_bucket=labels, labels_prefix="fishsense-lite"
    )

    async def _run():
        raw = await client.download_raw("deadbeef")
        await client.upload_processed_jpeg(
            folder="preprocess_groups_jpeg", checksum="cafef00d", data=b"JPG"
        )
        return raw

    assert await ActivityEnvironment().run(_run) == b"RAW"
    # JPEG lands in the labels bucket, under the prefix; NOT the scratch bucket.
    key = "fishsense-lite/preprocess_groups_jpeg/cafef00d.JPG"
    assert s3.get_object(Bucket=labels, Key=key)["Body"].read() == b"JPG"
    with pytest.raises(ClientError):
        s3.get_object(Bucket=BUCKET, Key=key)


@pytest.mark.asyncio
async def test_download_raw_raises_on_missing_key(s3):
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        return await client.download_raw("missing")

    with pytest.raises(ClientError) as exc_info:
        await ActivityEnvironment().run(_run)
    assert exc_info.value.response["Error"]["Code"] == "NoSuchKey"

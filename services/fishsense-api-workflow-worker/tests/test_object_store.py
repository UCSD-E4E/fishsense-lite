# pylint: disable=protected-access
"""Unit tests for the api-worker ObjectStoreClient (moto-backed).

Pins the cross-worker key contract and the staging/cleanup behavior:
  1. Key layout: raw/{checksum}.ORF, slate_pdf/{slate_id}.pdf.
  2. has_raw / has_slate_pdf reflect presence (HeadObject 404 → False).
  3. upload_raw / upload_slate_pdf round-trip bytes to the right key.
  4. delete_raw removes the raw scratch object and is idempotent
     (deleting an absent key is a success).
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker import object_store as sut

BUCKET = "fishsense-test"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _body(s3, key: str) -> bytes:
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


def _keys(s3) -> set[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET)
    return {o["Key"] for o in resp.get("Contents", [])}


# The key contract (`raw_key` / `slate_pdf_key` / `jpeg_key`) and
# `build_s3_client`'s path-style addressing now live in
# `fishsense_shared.object_store` and are pinned by
# `libs/fishsense-shared/tests/test_object_store.py`. Asserting them again
# here would be the same duplication this module was just refactored out of.
# What stays below is what is genuinely this worker's: its method subset.


async def test_upload_raw_writes_expected_key_and_bytes(s3):
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        await client.upload_raw("abc123", b"raw-bytes")

    await ActivityEnvironment().run(_run)

    assert _keys(s3) == {"raw/abc123.ORF"}
    assert _body(s3, "raw/abc123.ORF") == b"raw-bytes"


async def test_has_raw_reflects_presence(s3):
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _missing():
        return await client.has_raw("nope")

    async def _present():
        await client.upload_raw("yep", b"x")
        return await client.has_raw("yep")

    assert await ActivityEnvironment().run(_missing) is False
    assert await ActivityEnvironment().run(_present) is True


async def test_upload_slate_pdf_writes_expected_key(s3):
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        await client.upload_slate_pdf(42, b"%PDF-1.7")
        return await client.has_slate_pdf(42)

    present = await ActivityEnvironment().run(_run)
    assert present is True
    assert _keys(s3) == {"slate_pdf/42.pdf"}
    assert _body(s3, "slate_pdf/42.pdf") == b"%PDF-1.7"


async def test_delete_raw_removes_scratch_object_and_is_idempotent(s3):
    client = sut.ObjectStoreClient(s3, BUCKET)

    async def _run():
        await client.upload_raw("gone", b"x")
        first = await client.delete_raw("gone")
        # second delete on an already-absent key must still succeed
        second = await client.delete_raw("gone")
        return first, second

    first, second = await ActivityEnvironment().run(_run)
    assert first is True
    assert second is True
    assert _keys(s3) == set()


@pytest.mark.asyncio
async def test_has_processed_jpeg_checks_labels_bucket_and_prefix(s3):
    # JPEGs live in the labels bucket under labels_prefix; has_processed_jpeg
    # must look there, not in the scratch bucket.
    labels = "labels-fishsense-test"
    s3.create_bucket(Bucket=labels)
    client = sut.ObjectStoreClient(
        s3, BUCKET, labels_bucket=labels, labels_prefix="fishsense-lite"
    )

    async def _before():
        return await client.has_processed_jpeg("preprocess_groups_jpeg", "caf")

    assert await ActivityEnvironment().run(_before) is False

    # Writing it into the scratch bucket must NOT count.
    s3.put_object(
        Bucket=BUCKET, Key="fishsense-lite/preprocess_groups_jpeg/caf.JPG", Body=b"x"
    )
    assert await ActivityEnvironment().run(_before) is False

    # Only the labels bucket at the prefixed key counts.
    s3.put_object(
        Bucket=labels, Key="fishsense-lite/preprocess_groups_jpeg/caf.JPG", Body=b"x"
    )
    assert await ActivityEnvironment().run(_before) is True


@pytest.mark.asyncio
async def test_staging_uses_scratch_bucket_even_with_labels_configured(s3):
    # Raw staging always targets the scratch bucket, never the labels bucket.
    labels = "labels-fishsense-test"
    s3.create_bucket(Bucket=labels)
    client = sut.ObjectStoreClient(
        s3, BUCKET, labels_bucket=labels, labels_prefix="fishsense-lite"
    )

    async def _run():
        await client.upload_raw("abc123", b"RAW")
        return await client.has_raw("abc123")

    assert await ActivityEnvironment().run(_run) is True
    assert _body(s3, "raw/abc123.ORF") == b"RAW"
    # nothing written to the labels bucket
    assert s3.list_objects_v2(Bucket=labels).get("Contents", []) == []

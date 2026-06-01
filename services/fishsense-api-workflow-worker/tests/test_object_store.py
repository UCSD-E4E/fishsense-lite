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


def test_key_helpers():
    assert sut.raw_key("abc123") == "raw/abc123.ORF"
    assert sut.slate_pdf_key(7) == "slate_pdf/7.pdf"


def test_build_s3_client_uses_path_style_addressing_for_garage():
    """Garage has no virtual-host bucket DNS, so the client MUST use
    path-style addressing. A regression to virtual-host addressing
    would make every request hit `bucket.garage.example.com` and fail."""
    client = sut.build_s3_client(
        endpoint_url="http://garage.example.com",
        region="garage",
        access_key="k",
        secret_key="s",
    )
    assert client.meta.config.s3["addressing_style"] == "path"
    assert client.meta.endpoint_url == "http://garage.example.com"


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

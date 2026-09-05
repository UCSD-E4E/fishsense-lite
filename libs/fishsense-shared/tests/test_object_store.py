# pylint: disable=protected-access
"""Unit tests for the shared Garage object-store contract.

This module is the single source of truth for the cross-worker key
contract and the S3 client shape, so the things pinned here are exactly
the things that used to be able to drift between the api-worker's and
data-worker's private copies:

  1. Key layout — raw/{checksum}.ORF, slate_pdf/{slate_id}.pdf,
     {prefix}/{folder}/{checksum}.JPG.
  2. Path-style addressing (Garage has no virtual-host bucket DNS).
  3. `BaseObjectStoreClient`'s bucket routing: scratch reads/writes go to
     `bucket`, label-facing JPEGs to `labels_bucket` under `labels_prefix`.
  4. `_get` closes the StreamingBody. Only the data-worker's copy did
     this; the api-worker's leaked the connection back into botocore's
     pool. Pinned here so one implementation can't regress alone.
"""

from __future__ import annotations

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from fishsense_shared import object_store as sut

BUCKET = "fishsense-test"
LABELS_BUCKET = "labels-fishsense-test"


@pytest.fixture(name="s3")
def s3_fixture():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        client.create_bucket(Bucket=LABELS_BUCKET)
        yield client


# --------------------------------------------------------------------
# Key contract
# --------------------------------------------------------------------


def test_raw_and_slate_pdf_keys():
    assert sut.raw_key("deadbeef") == "raw/deadbeef.ORF"
    assert sut.slate_pdf_key(9) == "slate_pdf/9.pdf"
    assert sut.RAW_PREFIX == "raw"
    assert sut.SLATE_PDF_PREFIX == "slate_pdf"


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("", "preprocess_jpeg/abc.JPG"),
        (None, "preprocess_jpeg/abc.JPG"),
        ("fishsense-lite", "fishsense-lite/preprocess_jpeg/abc.JPG"),
        # Leading/trailing slashes are stripped so callers can't produce
        # a double-slash key that reads as a different object.
        ("/fishsense-lite/", "fishsense-lite/preprocess_jpeg/abc.JPG"),
    ],
)
def test_jpeg_key_prefix_handling(prefix, expected):
    assert sut.jpeg_key("preprocess_jpeg", "abc", prefix) == expected


@pytest.mark.parametrize(
    "folder",
    [
        "preprocess_jpeg",
        "preprocess_groups_jpeg",
        "preprocess_headtail_jpeg",
        "preprocess_slate_images_jpeg",
    ],
)
def test_jpeg_key_covers_every_stage_folder(folder):
    assert sut.jpeg_key(folder, "cafef00d") == f"{folder}/cafef00d.JPG"


def test_build_s3_client_uses_path_style_addressing_for_garage():
    """Garage has no virtual-host bucket DNS. A regression to virtual-host
    addressing would send every request to `bucket.garage.example.com`."""
    client = sut.build_s3_client(
        endpoint_url="http://garage.example.com",
        region="garage",
        access_key="k",
        secret_key="s",
    )
    assert client.meta.config.s3["addressing_style"] == "path"
    assert client.meta.config.signature_version == "s3v4"
    assert client.meta.endpoint_url == "http://garage.example.com"


# --------------------------------------------------------------------
# Bucket routing
# --------------------------------------------------------------------


def test_labels_bucket_defaults_to_scratch_bucket():
    """Single-bucket layouts keep working: an unset labels_bucket means
    JPEGs land in the same bucket as scratch."""
    client = sut.BaseObjectStoreClient(object(), BUCKET)
    assert client._labels_bucket == BUCKET
    assert client._labels_prefix == ""


def test_labels_bucket_and_prefix_are_honored_when_set():
    client = sut.BaseObjectStoreClient(
        object(), BUCKET, labels_bucket=LABELS_BUCKET, labels_prefix="fishsense-lite"
    )
    assert client._labels_bucket == LABELS_BUCKET
    assert client._labels_prefix == "fishsense-lite"


def test_none_labels_prefix_normalizes_to_empty_string():
    client = sut.BaseObjectStoreClient(object(), BUCKET, labels_prefix=None)
    assert client._labels_prefix == ""


# --------------------------------------------------------------------
# _exists / _get / _put / _delete
# --------------------------------------------------------------------


async def test_exists_true_when_object_present(s3):
    s3.put_object(Bucket=BUCKET, Key="raw/abc.ORF", Body=b"x")
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    assert await client._exists("raw/abc.ORF") is True


async def test_exists_false_when_object_missing(s3):
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    assert await client._exists("raw/nope.ORF") is False


async def test_exists_reraises_non_not_found_errors(s3):
    """A 403/500 must not be silently reported as "absent" — that would
    make the staging activity re-upload on every firing, or worse, make
    cleanup believe it had nothing to delete."""

    class _Boom:
        def head_object(self, **_kwargs):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "nope"}}, "HeadObject"
            )

    client = sut.BaseObjectStoreClient(_Boom(), BUCKET)
    with pytest.raises(ClientError) as exc_info:
        await client._exists("raw/abc.ORF")
    assert exc_info.value.response["Error"]["Code"] == "AccessDenied"
    _ = s3  # bucket fixture keeps moto active for symmetry


async def test_exists_checks_the_override_bucket(s3):
    s3.put_object(Bucket=LABELS_BUCKET, Key="preprocess_jpeg/abc.JPG", Body=b"j")
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    assert await client._exists("preprocess_jpeg/abc.JPG", bucket=LABELS_BUCKET) is True
    # Same key, default (scratch) bucket -> absent.
    assert await client._exists("preprocess_jpeg/abc.JPG") is False


async def test_get_returns_bytes_and_closes_the_streaming_body():
    """botocore hands back a StreamingBody that holds an HTTP connection.
    Leaking it across repeated downloads exhausts the pool and stalls the
    activity — the data-worker fixed this, the api-worker didn't."""
    closed: list[bool] = []

    class _Body:
        def read(self):
            return b"PAYLOAD"

        def close(self):
            closed.append(True)

    class _S3:
        def get_object(self, **_kwargs):
            return {"Body": _Body()}

    client = sut.BaseObjectStoreClient(_S3(), BUCKET)
    assert await client._get("raw/abc.ORF") == b"PAYLOAD"
    assert closed == [True], "StreamingBody was not closed"


async def test_get_reads_from_the_override_bucket(s3):
    s3.put_object(Bucket=LABELS_BUCKET, Key="k", Body=b"LABELS")
    s3.put_object(Bucket=BUCKET, Key="k", Body=b"SCRATCH")
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    assert await client._get("k") == b"SCRATCH"
    assert await client._get("k", bucket=LABELS_BUCKET) == b"LABELS"


async def test_get_raises_on_missing_key(s3):
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    with pytest.raises(ClientError) as exc_info:
        await client._get("raw/missing.ORF")
    assert exc_info.value.response["Error"]["Code"] == "NoSuchKey"


async def test_put_writes_to_the_default_and_override_buckets(s3):
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    await client._put("raw/abc.ORF", b"RAW")
    await client._put("preprocess_jpeg/abc.JPG", b"JPG", bucket=LABELS_BUCKET)
    assert s3.get_object(Bucket=BUCKET, Key="raw/abc.ORF")["Body"].read() == b"RAW"
    assert (
        s3.get_object(Bucket=LABELS_BUCKET, Key="preprocess_jpeg/abc.JPG")[
            "Body"
        ].read()
        == b"JPG"
    )


async def test_delete_removes_the_object_and_is_idempotent(s3):
    s3.put_object(Bucket=BUCKET, Key="raw/abc.ORF", Body=b"x")
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    await client._delete("raw/abc.ORF")
    assert "Contents" not in s3.list_objects_v2(Bucket=BUCKET)
    # S3 delete_object on an absent key succeeds, so retries are safe.
    await client._delete("raw/abc.ORF")


async def test_delete_targets_the_override_bucket(s3):
    s3.put_object(Bucket=LABELS_BUCKET, Key="k", Body=b"x")
    s3.put_object(Bucket=BUCKET, Key="k", Body=b"y")
    client = sut.BaseObjectStoreClient(s3, BUCKET)
    await client._delete("k", bucket=LABELS_BUCKET)
    assert "Contents" not in s3.list_objects_v2(Bucket=LABELS_BUCKET)
    assert s3.get_object(Bucket=BUCKET, Key="k")["Body"].read() == b"y"


# --------------------------------------------------------------------
# Settings -> client factory
# --------------------------------------------------------------------


class _FakeObjectStoreSettings:
    """Stands in for Dynaconf's `settings.object_store`: attribute access
    for required keys, `.get()` with a default for optional ones."""

    def __init__(self, **values):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeSettings:
    def __init__(self, object_store):
        self.object_store = object_store


def _settings(**overrides):
    base = {
        "endpoint_url": "http://garage.example.com",
        "region": "garage",
        "access_key": "k",
        "secret_key": "s",
        "bucket": BUCKET,
    }
    base.update(overrides)
    return _FakeSettings(_FakeObjectStoreSettings(**base))


def test_open_client_builds_the_requested_subclass():
    """Both workers call this with their own ObjectStoreClient subclass —
    the factory must not hardcode a class."""

    class _Marker(sut.BaseObjectStoreClient):
        pass

    client = sut.open_client(_settings(), _Marker)
    assert isinstance(client, _Marker)
    assert client._bucket == BUCKET


def test_open_client_defaults_optional_labels_settings():
    """A single-bucket deployment sets neither key; both must fall back
    without raising, since Dynaconf has no such attributes to read."""
    client = sut.open_client(_settings(), sut.BaseObjectStoreClient)
    assert client._labels_bucket == BUCKET
    assert client._labels_prefix == ""


def test_open_client_honors_split_bucket_settings():
    client = sut.open_client(
        _settings(labels_bucket=LABELS_BUCKET, labels_prefix="fishsense-lite"),
        sut.BaseObjectStoreClient,
    )
    assert client._labels_bucket == LABELS_BUCKET
    assert client._labels_prefix == "fishsense-lite"


def test_open_client_normalizes_a_null_labels_prefix():
    """`labels_prefix = ""` in TOML round-trips as None on some Dynaconf
    paths; it must not become the string "None" in a key."""
    client = sut.open_client(
        _settings(labels_prefix=None), sut.BaseObjectStoreClient
    )
    assert client._labels_prefix == ""


def test_open_client_passes_garage_addressing_through():
    client = sut.open_client(_settings(), sut.BaseObjectStoreClient)
    assert client._s3.meta.config.s3["addressing_style"] == "path"
    assert client._s3.meta.endpoint_url == "http://garage.example.com"


def test_model_key_is_versioned():
    """Model checkpoints are addressed `models/{name}/{version}/{filename}`.

    The version is part of the key on purpose. Checkpoints are cached on a
    volume by this same path, so if two sets of weights could share a key a
    cold start could silently serve the wrong one — and the cache would keep
    serving it until someone cleared the volume by hand.

    This is a cross-worker key contract like the JPEG prefixes: the data-worker
    reads it and whoever uploads the weights writes it, so it is pinned here
    rather than left to agree by convention.
    """
    assert (
        sut.model_key("sam3", "v1", "sam3.pt") == f"{sut.MODEL_PREFIX}/sam3/v1/sam3.pt"
    )


def test_model_key_separates_versions_of_the_same_model():
    assert sut.model_key("sam3", "v1", "w.pt") != sut.model_key("sam3", "v2", "w.pt")

# pylint: disable=unused-argument
"""Unit tests for cleanup_raw_bytes_for_dive_activity.

Cleanup deletes only the Garage `raw/{checksum}.ORF` SCRATCH objects.
The NAS source is never touched — the activity has no NAS client at all,
which these tests assert structurally.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import boto3
import pytest
from moto import mock_aws
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    cleanup_raw_bytes_for_dive_activity as sut,
)
from fishsense_api_workflow_worker.object_store import ObjectStoreClient, raw_key

BUCKET = "fishsense-test"


def _image(image_id: int, *, checksum: str = "abc") -> Image:
    return Image(
        id=image_id,
        path="/share/x.ORF",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=checksum,
        is_canonical=True,
        dive_id=42,
        camera_id=1,
    )


def _make_fs(images: List[Image]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    return fs


@contextmanager
def _moto_store(monkeypatch, *, preexisting: tuple[str, ...] = ()):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        for checksum in preexisting:
            s3.put_object(Bucket=BUCKET, Key=raw_key(checksum), Body=b"raw")
        client = ObjectStoreClient(s3, BUCKET)
        monkeypatch.setattr(sut, "open_object_store_client", lambda: client)
        # Cleanup now asks Temporal whether another stage's child is still
        # reading this dive's scratch (see `scratch_in_use`). These tests are
        # about the delete itself, so the scratch is declared free; the gate has
        # its own tests below and in
        # `test_cleanup_raw_respects_other_children.py`.
        monkeypatch.setattr(sut, "scratch_in_use", AsyncMock(return_value=None))
        yield s3


def _raw_keys(s3) -> set[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="raw/")
    return {o["Key"] for o in resp.get("Contents", [])}


@pytest.mark.asyncio
async def test_deletes_all_raw_orfs_for_dive(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa"), _image(2, checksum="bbb")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with _moto_store(monkeypatch, preexisting=("aaa", "bbb")) as s3:
        result = await ActivityEnvironment().run(
            sut.cleanup_raw_bytes_for_dive_activity, 42
        )

        assert result.deleted == 2
        # both scratch objects gone from Garage
        assert _raw_keys(s3) == set()


@pytest.mark.asyncio
async def test_delete_of_absent_key_counted_as_success(monkeypatch):
    fs = _make_fs([_image(1, checksum="aaa")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    # nothing pre-seeded: the key is already absent. S3 delete_object is
    # idempotent, so this still counts as a successful no-op (retried
    # cleanups don't raise).
    with _moto_store(monkeypatch):
        result = await ActivityEnvironment().run(
            sut.cleanup_raw_bytes_for_dive_activity, 42
        )

    assert result.deleted == 1


@pytest.mark.asyncio
async def test_returns_zero_for_empty_dive(monkeypatch):
    fs = _make_fs([])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with _moto_store(monkeypatch):
        result = await ActivityEnvironment().run(
            sut.cleanup_raw_bytes_for_dive_activity, 42
        )

    assert result.deleted == 0


def test_activity_module_imports_no_nas_client():
    """Structural guard for the NAS-safety invariant: cleanup must not
    import or hold any NAS client. If a future change wires the NAS
    client module into cleanup, this test fails — a deliberate tripwire
    so nobody accidentally adds a NAS-delete path to the cleanup
    activity."""
    import inspect

    source = inspect.getsource(sut)
    assert "fishsense_api_workflow_worker.nas" not in source, (
        "cleanup_raw_bytes_for_dive_activity imported the NAS client — "
        "cleanup must only ever delete the Garage scratch copy, never the "
        "NAS source."
    )
    assert "NasClient" not in source and "NasDownloadClient" not in source


@pytest.mark.asyncio
async def test_does_not_delete_while_another_stage_is_still_reading(monkeypatch):
    """Scratch is keyed per dive, so a sibling stage's child may still need it.

    Prod dive 442, 2026-09-07: the species parent's cleanup deleted 984 objects
    while the laser child was mid-render, and that child died with NoSuchKey.
    """
    fs = _make_fs([_image(1, checksum="aaa"), _image(2, checksum="bbb")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    with _moto_store(monkeypatch, preexisting=("aaa", "bbb")) as s3:
        monkeypatch.setattr(
            sut, "scratch_in_use", AsyncMock(return_value="preprocess-laser-42")
        )
        result = await ActivityEnvironment().run(
            sut.cleanup_raw_bytes_for_dive_activity, 42
        )
        assert result.deleted == 0
        assert _raw_keys(s3) == {raw_key("aaa"), raw_key("bbb")}, "nothing deleted"


@pytest.mark.asyncio
async def test_an_unreachable_temporal_reads_as_in_use(monkeypatch):
    """Fails closed. The two ways of being wrong are not symmetric: deleting
    under a live child kills a render silently and costs the dive's whole NAS
    staging to redo, while keeping the scratch costs object-store space until
    the next firing re-stages -- which reports `skipped_already_present` and
    reuses what is there.

    Calls the real `scratch_in_use`, not the fixture's stub.
    """
    monkeypatch.setattr(
        sut.Client, "connect", AsyncMock(side_effect=RuntimeError("dns error"))
    )
    assert await sut.scratch_in_use(42) is not None, (
        "an unknown answer must block the delete, not wave it through"
    )

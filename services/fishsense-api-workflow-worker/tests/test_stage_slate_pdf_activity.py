# pylint: disable=unused-argument
"""Unit tests for stage_slate_pdf_activity.

Storage is exercised against moto (a real boto3 client over an
in-memory S3); the slate PDF lands at the `slate_pdf/{slate_id}.pdf`
Garage key.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import List
from unittest.mock import AsyncMock, MagicMock

import boto3
import pytest
from moto import mock_aws
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate import DiveSlate
from fishsense_api_workflow_worker.activities import stage_slate_pdf_activity as sut
from fishsense_api_workflow_worker.object_store import (
    ObjectStoreClient,
    slate_pdf_key,
)

BUCKET = "fishsense-test"


def _slate(
    slate_id: int = 7, *, path: str = "slates/v1/slate.pdf"
) -> DiveSlate:
    """Build a fake DiveSlate. Default `path` is **share-relative**
    (no leading slash) — matches prod DB shape. Tests using this
    default exercise the activity's `_resolve_nas_path` prefixing;
    reverting that prefixing breaks them."""
    return DiveSlate(
        id=slate_id,
        name="test",
        dpi=300,
        path=path,
        created_at=None,
        reference_points=[(0.0, 0.0)],
    )


def _make_fs(slates: List[DiveSlate]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(return_value=slates)
    return fs


@contextmanager
def _moto_store(monkeypatch, *, preexisting_slate_ids: tuple[int, ...] = ()):
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        for slate_id in preexisting_slate_ids:
            s3.put_object(
                Bucket=BUCKET, Key=slate_pdf_key(slate_id), Body=b"old"
            )
        client = ObjectStoreClient(s3, BUCKET)
        monkeypatch.setattr(sut, "open_object_store_client", lambda: client)
        yield s3


def _slate_keys(s3) -> set[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="slate_pdf/")
    return {o["Key"] for o in resp.get("Contents", [])}


def _patch_tempfile_read(monkeypatch, payload: bytes):
    real_path = sut.Path

    class _PathStub(real_path):  # type: ignore[misc]
        def read_bytes(self):  # type: ignore[override]
            return payload

    monkeypatch.setattr(sut, "Path", _PathStub)


@pytest.mark.asyncio
async def test_skips_nas_when_pdf_already_present(monkeypatch):
    fs = _make_fs([_slate(7)])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    nas = MagicMock()
    nas.download_to = MagicMock()
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch, preexisting_slate_ids=(7,)) as s3:
        result = await ActivityEnvironment().run(
            sut.stage_slate_pdf_activity, 7
        )

        assert result is True
        nas.download_to.assert_not_called()
        # already-present PDF is not re-written
        assert (
            s3.get_object(Bucket=BUCKET, Key=slate_pdf_key(7))["Body"].read()
            == b"old"
        )


@pytest.mark.asyncio
async def test_downloads_and_puts_when_pdf_missing(monkeypatch):
    """The default `_slate()` path is share-relative; the activity
    must rewrite it to absolute before passing to the NAS client.
    Reverting `_resolve_nas_path` would make this assertion fail."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    fs = _make_fs([_slate(7)])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    nas = MagicMock()
    nas.download_to = MagicMock()
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"pdf-bytes")

    with _moto_store(monkeypatch) as s3:
        result = await ActivityEnvironment().run(
            sut.stage_slate_pdf_activity, 7
        )

        assert result is True
        nas.download_to.assert_called_once()
        assert nas.download_to.call_args.kwargs["src_path"] == (
            "/fishsense_data/REEF/data/slates/v1/slate.pdf"
        )
        # And explicitly NOT the bare DB path — regression guard.
        assert (
            nas.download_to.call_args.kwargs["src_path"] != "slates/v1/slate.pdf"
        )
        assert _slate_keys(s3) == {slate_pdf_key(7)}
        assert (
            s3.get_object(Bucket=BUCKET, Key=slate_pdf_key(7))["Body"].read()
            == b"pdf-bytes"
        )


@pytest.mark.asyncio
async def test_raises_when_slate_missing(monkeypatch):
    fs = _make_fs([_slate(7)])  # only id=7 exists
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="not found"):
        await ActivityEnvironment().run(sut.stage_slate_pdf_activity, 99)


@pytest.mark.asyncio
async def test_raises_when_slate_path_is_empty(monkeypatch):
    fs = _make_fs([_slate(7, path="")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="no NAS path"):
        await ActivityEnvironment().run(sut.stage_slate_pdf_activity, 7)


@pytest.mark.asyncio
async def test_relative_slate_path_gets_prefixed_before_nas_download(monkeypatch):
    """Mirrors `stage_raw_bytes_for_dive_activity`: a share-relative
    `dive_slate.path` gets `e4e_nas.raw_root_path` prepended before
    reaching FileStation."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    fs = _make_fs([_slate(7, path="slates/v1/slate.pdf")])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    nas = MagicMock()
    nas.download_to = MagicMock()
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"pdf-bytes")

    with _moto_store(monkeypatch):
        result = await ActivityEnvironment().run(sut.stage_slate_pdf_activity, 7)

    assert result is True
    nas.download_to.assert_called_once()
    assert nas.download_to.call_args.kwargs["src_path"] == (
        "/fishsense_data/REEF/data/slates/v1/slate.pdf"
    )

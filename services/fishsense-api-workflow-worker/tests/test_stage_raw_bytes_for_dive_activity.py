# pylint: disable=unused-argument
"""Unit tests for stage_raw_bytes_for_dive_activity.

Pins down:
  1. HEAD-skips already-staged checksums (no NAS download, no PUT).
  2. Newly-staged images: NAS download → object-store PUT.
  3. Images without `path` or `checksum` get counted as `no_path`
     and don't crash.
  4. Returns the per-dive summary so the parent workflow can log
     counts.
  5. Share-relative `image.path` values (the DB convention) get
     `e4e_nas.raw_root_path` prepended before being handed to
     FileStation; absolute paths pass through unchanged.
  6. (2026-05-07 stage 2 incident invariant) When the NAS download
     fails, the activity raises without uploading anything to the
     object store. This is the "don't propagate corrupt content"
     guard — even if the underlying NAS client ever regressed to
     produce empty/JSON content, the activity must not turn around
     and PUT it as `.ORF`.

Storage is exercised against moto (a real boto3 client over an
in-memory S3), mirroring how the data-worker's object-store tests run
and how the prior httpx.MockTransport exercised the nginx exchange.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List
from unittest.mock import AsyncMock, MagicMock

import boto3
import pytest
from moto import mock_aws
from synology_filestation import DSMError, TransportError
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.image import Image
from fishsense_api_workflow_worker.activities import (
    stage_raw_bytes_for_dive_activity as sut,
)
from fishsense_api_workflow_worker.object_store import ObjectStoreClient, raw_key

BUCKET = "fishsense-test"


def _image(
    image_id: int,
    *,
    path: str = "2024.06.20.REEF/dive_42/IMG.ORF",
    checksum: str = "abc",
) -> Image:
    """Build a fake Image. Default `path` is **share-relative** (no
    leading slash) — same shape the prod DB stores. This makes every
    test that uses the default exercise the activity's
    `_resolve_nas_path` prefixing. Reverting that prefixing would make
    the existing tests fail because the NAS would receive bare
    relative paths it can't resolve."""
    return Image(
        id=image_id,
        path=path,
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


def _make_nas() -> MagicMock:
    nas = MagicMock()
    # download_to writes a file to dest_dir; simulate with a side_effect.
    nas.download_to = MagicMock()
    return nas


@contextmanager
def _moto_store(monkeypatch, *, preexisting: tuple[str, ...] = ()):
    """Stand up an in-memory S3 bucket, pre-seed `preexisting` raw
    checksums, and patch the activity's object-store factory to return
    a client bound to it. Yields the raw boto3 client for assertions."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=BUCKET)
        for checksum in preexisting:
            s3.put_object(Bucket=BUCKET, Key=raw_key(checksum), Body=b"old")
        client = ObjectStoreClient(s3, BUCKET)
        monkeypatch.setattr(sut, "open_object_store_client", lambda: client)
        yield s3


def _raw_keys(s3) -> set[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix="raw/")
    return {o["Key"] for o in resp.get("Contents", [])}


def _patch_tempfile_read(monkeypatch, payload: bytes):
    """Make every Path(...).read_bytes() return `payload` so the test
    doesn't need to actually drop a real file on disk."""
    real_path = sut.Path

    class _PathStub(real_path):  # type: ignore[misc]
        def read_bytes(self):  # type: ignore[override]
            return payload

    monkeypatch.setattr(sut, "Path", _PathStub)


@pytest.mark.asyncio
async def test_skips_already_staged_checksums_no_nas_download(monkeypatch):
    images = [_image(1, checksum="aaa"), _image(2, checksum="bbb")]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch, preexisting=("aaa", "bbb")) as s3:
        result = await ActivityEnvironment().run(
            sut.stage_raw_bytes_for_dive_activity, 42
        )

        assert result.staged == 0
        assert result.skipped_already_present == 2
        assert result.no_path == 0
        nas.download_to.assert_not_called()
        # nothing re-written: both keys keep their original bytes
        assert _raw_keys(s3) == {raw_key("aaa"), raw_key("bbb")}
        assert (
            s3.get_object(Bucket=BUCKET, Key=raw_key("aaa"))["Body"].read()
            == b"old"
        )


@pytest.mark.asyncio
async def test_stages_new_checksums_via_nas_download_then_put(monkeypatch):
    """The default `_image()` path is share-relative
    (`2024.06.20.REEF/dive_42/IMG.ORF`); the activity must rewrite
    it to absolute (`/fishsense_data/REEF/data/...`) before passing
    it to the NAS client. Reverting `_resolve_nas_path` makes this
    assertion fail.
    """
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    images = [_image(1, checksum="aaa")]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"raw-bytes")

    with _moto_store(monkeypatch) as s3:
        result = await ActivityEnvironment().run(
            sut.stage_raw_bytes_for_dive_activity, 42
        )

        assert result.staged == 1
        assert result.skipped_already_present == 0
        assert nas.download_to.call_count == 1
        nas_call = nas.download_to.call_args
        assert nas_call.kwargs["src_path"] == (
            "/fishsense_data/REEF/data/2024.06.20.REEF/dive_42/IMG.ORF"
        )
        # And explicitly NOT the bare DB path — this is the regression
        # guard: a future "just pass image.path through" change must fail
        # this test.
        assert nas_call.kwargs["src_path"] != "2024.06.20.REEF/dive_42/IMG.ORF"
        assert _raw_keys(s3) == {raw_key("aaa")}
        assert (
            s3.get_object(Bucket=BUCKET, Key=raw_key("aaa"))["Body"].read()
            == b"raw-bytes"
        )


@pytest.mark.asyncio
async def test_counts_no_path_images_without_crashing(monkeypatch):
    images = [
        _image(1, path="", checksum="aaa"),
        _image(2, path="dive_x/file.ORF", checksum=""),
        _image(3, path="dive_y/file.ORF", checksum="ccc"),
    ]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"raw")

    with _moto_store(monkeypatch):
        result = await ActivityEnvironment().run(
            sut.stage_raw_bytes_for_dive_activity, 42
        )

    assert result.no_path == 2
    assert result.staged == 1


@pytest.mark.asyncio
async def test_failed_nas_download_does_not_upload_to_object_store(monkeypatch):
    """Invariant for the 2026-05-07 stage 2 incident.

    The original failure mode was: synology-api silently produced a
    non-`.ORF` file in the activity's tempdir (a JSON-error response
    body), `read_bytes()` returned those bytes, the staging upload PUT
    them. Migrating to `synology-filestation` makes the underlying
    client raise on JSON-error — but the activity-side invariant should
    hold independently of which client is in use: if the download path
    fails for any reason, no object is written.

    This test wires the NAS client to raise during `download_to` and
    asserts (a) the activity surfaces the failure to the caller, and
    (b) no checksum object is written. A future regression that catches
    and ignores the download exception (or that stages fallback bytes)
    would re-trip this alarm before any corrupt content could reach
    prod.
    """
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    images = [_image(1, checksum="aaa")]
    fs = _make_fs(images)
    nas = _make_nas()
    nas.download_to.side_effect = RuntimeError(
        "simulated DSM session expired (code 119)"
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch) as s3:
        with pytest.raises(Exception):
            await ActivityEnvironment().run(
                sut.stage_raw_bytes_for_dive_activity, 42
            )

        assert nas.download_to.call_count == 1
        assert _raw_keys(s3) == set(), (
            "stage_raw_bytes_for_dive_activity uploaded to the object store "
            "after the NAS download raised — this is exactly the failure "
            "shape that propagated DSM JSON-error bodies as `.ORF` content "
            "on 2026-05-07."
        )


@pytest.mark.asyncio
async def test_returns_zeros_when_dive_has_no_images(monkeypatch):
    fs = _make_fs([])
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch):
        result = await ActivityEnvironment().run(
            sut.stage_raw_bytes_for_dive_activity, 42
        )

    assert result.staged == 0
    assert result.skipped_already_present == 0
    assert result.no_path == 0
    nas.download_to.assert_not_called()


def test_resolve_nas_path_prepends_root_to_relative_paths(monkeypatch):
    """Share-relative paths (the DB convention) get the configured root
    prepended; absolute paths are returned unchanged."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    assert (
        sut._resolve_nas_path(  # pylint: disable=protected-access
            "2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF"
        )
        == "/fishsense_data/REEF/data/2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF"
    )
    # Absolute path: passthrough so an operator override or a future
    # path migration isn't double-prefixed.
    assert (
        sut._resolve_nas_path("/already/absolute/file.ORF")  # pylint: disable=protected-access
        == "/already/absolute/file.ORF"
    )
    # Trailing slash on root is normalized.
    monkeypatch.setenv("E4EFS_E4E_NAS__RAW_ROOT_PATH", "/foo/bar/")
    cfg.settings.reload()
    assert (
        sut._resolve_nas_path("rel/path.ORF")  # pylint: disable=protected-access
        == "/foo/bar/rel/path.ORF"
    )


@pytest.mark.asyncio
async def test_relative_image_paths_get_prefixed_before_nas_download(monkeypatch):
    """End-to-end: a share-relative `image.path` is rewritten with the
    root prefix before reaching the NAS client. Mirrors the prod path
    shape — DB stores `2024.06.20.REEF/.../P8290052.ORF`, NAS expects
    `/fishsense_data/REEF/data/2024.06.20.REEF/.../P8290052.ORF`."""
    monkeypatch.setenv(
        "E4EFS_E4E_NAS__RAW_ROOT_PATH", "/fishsense_data/REEF/data"
    )
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    images = [
        _image(
            1,
            path="2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF",
            checksum="aaa",
        )
    ]
    fs = _make_fs(images)
    nas = _make_nas()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)
    _patch_tempfile_read(monkeypatch, b"raw-bytes")

    with _moto_store(monkeypatch) as s3:
        result = await ActivityEnvironment().run(
            sut.stage_raw_bytes_for_dive_activity, 42
        )

        assert result.staged == 1
        nas.download_to.assert_called_once()
        assert nas.download_to.call_args.kwargs["src_path"] == (
            "/fishsense_data/REEF/data/2024.06.20.REEF/08_2023/082929_FishModels_FSL04/P8290052.ORF"
        )
        # Object-store PUT still keys on checksum, not the rewritten path.
        assert _raw_keys(s3) == {raw_key("aaa")}


@pytest.mark.asyncio
async def test_transient_502_propagates_without_inner_retry(monkeypatch):
    """A 502 (TransportError) fails the activity so Temporal's bounded,
    backed-off retry_policy reschedules it — the activity itself does NOT
    loop internally (that inner-under-outer retry is what produced the
    storm). One attempt per activity execution; nothing staged."""
    images = [_image(1, checksum="aaa")]
    fs = _make_fs(images)
    nas = _make_nas()
    nas.download_to.side_effect = TransportError("download HTTP 502 Bad Gateway")
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch) as s3:
        with pytest.raises(BaseException) as excinfo:
            await ActivityEnvironment().run(
                sut.stage_raw_bytes_for_dive_activity, 42
            )
        # transient -> NOT surfaced as a non-retryable ApplicationError
        leaves = list(sut._iter_leaf_exceptions(excinfo.value))  # pylint: disable=protected-access
        assert not any(
            isinstance(leaf, ApplicationError) and leaf.non_retryable
            for leaf in leaves
        )
        assert nas.download_to.call_count == 1  # no inner retry loop
        assert _raw_keys(s3) == set()


@pytest.mark.asyncio
async def test_permanent_408_raises_non_retryable_application_error(monkeypatch):
    """Synology 408 ("no such file") is permanent — surfaced un-wrapped as
    a non-retryable ApplicationError so Temporal fails the firing fast
    instead of burning the retry budget on a doomed staging. The file is
    never skipped, so the missing raw is not silently dropped."""
    images = [_image(1, checksum="aaa")]
    fs = _make_fs(images)
    nas = _make_nas()
    nas.download_to.side_effect = DSMError("Synology API error 408")
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch) as s3:
        with pytest.raises(ApplicationError) as excinfo:
            await ActivityEnvironment().run(
                sut.stage_raw_bytes_for_dive_activity, 42
            )
        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == sut.NAS_FILE_NOT_FOUND_TYPE
        assert nas.download_to.call_count == 1
        assert _raw_keys(s3) == set()


@pytest.mark.asyncio
async def test_transient_dsm_407_propagates_retryable(monkeypatch):
    """407 was the backend fail-closing during the incident — transient,
    so it must NOT be classified permanent; it propagates for Temporal to
    back off and retry."""
    images = [_image(1, checksum="aaa")]
    fs = _make_fs(images)
    nas = _make_nas()
    nas.download_to.side_effect = DSMError("Synology API error 407")
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_build_nas_client", lambda: nas)

    with _moto_store(monkeypatch):
        with pytest.raises(BaseException) as excinfo:
            await ActivityEnvironment().run(
                sut.stage_raw_bytes_for_dive_activity, 42
            )
        leaves = list(sut._iter_leaf_exceptions(excinfo.value))  # pylint: disable=protected-access
        assert not any(
            isinstance(leaf, ApplicationError) and leaf.non_retryable
            for leaf in leaves
        )


def test_stage_raw_retry_policy_is_bounded_and_marks_missing_file_non_retryable():
    """Contract between the activity and the parents' retry policy: the
    non-retryable ApplicationError `type` the activity raises for a
    permanent (408) error must be exactly what the policy lists in
    `non_retryable_error_types`, and attempts must be bounded (no storm)."""
    from fishsense_api_workflow_worker.workflows._retry_policies import (  # pylint: disable=import-outside-toplevel
        STAGE_RAW_RETRY_POLICY,
    )

    assert STAGE_RAW_RETRY_POLICY.maximum_attempts == 5
    assert sut.NAS_FILE_NOT_FOUND_TYPE in (
        STAGE_RAW_RETRY_POLICY.non_retryable_error_types or []
    )

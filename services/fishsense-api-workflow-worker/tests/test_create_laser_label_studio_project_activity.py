"""Unit tests for create_laser_label_studio_project_activity.

Exercises both branches of `create_or_get_label_studio_project`:
title-based idempotent return when a project already exists, and
create-from-XML when it doesn't. The "missing XML" path must raise
rather than create an unlabel-able empty project.

The same `create_or_get_label_studio_project` helper backs all four
stages, so this single test file covers the common code; the
species/headtail/dive_slate variants are thin wrappers and would
just duplicate this coverage.

The activity now takes a `dive_id` and constructs a per-dive title
of the form `"{dive.name} - Laser Calibration Labeling"` via
`build_per_dive_title`. The dive lookup is mocked here so the unit
tests don't need a real fishsense-api.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    create_laser_label_studio_project_activity as sut,
    populate_utils as sut_utils,
)


def _make_ls(*, list_result=None, create_result=None):
    ls = MagicMock()
    ls.projects = MagicMock()
    ls.projects.list = MagicMock(return_value=list_result or [])
    ls.projects.create = MagicMock(return_value=create_result)
    return ls


def _patch_dive_lookup(monkeypatch, *, dive_name: str | None):
    """Stub `populate_utils.get_fs_client` so `build_per_dive_title`
    sees a deterministic dive."""

    fake_dive = SimpleNamespace(id=393, name=dive_name)

    fake_fs = MagicMock()

    async def _get(**_kwargs):
        return fake_dive

    fake_fs.dives.get = _get

    @asynccontextmanager
    async def _client():
        yield fake_fs

    monkeypatch.setattr(sut_utils, "get_fs_client", _client)


@pytest.mark.asyncio
async def test_returns_existing_project_id_by_title_match(monkeypatch):
    _patch_dive_lookup(monkeypatch, dive_name="dive-alpha")
    expected_title = "dive-alpha #393 - Laser Calibration Labeling"

    ls = _make_ls(
        list_result=[
            SimpleNamespace(id=42, title="Random Other Project"),
            SimpleNamespace(id=73, title=expected_title),
        ]
    )
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(
        sut.create_laser_label_studio_project_activity, 393
    )

    assert result == 73
    ls.projects.create.assert_not_called()


@pytest.mark.asyncio
async def test_creates_project_when_no_match_and_xml_present(monkeypatch):
    _patch_dive_lookup(monkeypatch, dive_name="dive-alpha")
    expected_title = "dive-alpha #393 - Laser Calibration Labeling"
    monkeypatch.setattr(sut, "LASER_LABELING_CONFIG_XML", "<View><Image/></View>")

    ls = _make_ls(
        list_result=[],
        create_result=SimpleNamespace(id=101, title=expected_title),
    )
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(
        sut.create_laser_label_studio_project_activity, 393
    )

    assert result == 101
    ls.projects.create.assert_called_once_with(
        title=expected_title,
        label_config="<View><Image/></View>",
        workspace=None,
    )


@pytest.mark.asyncio
async def test_raises_when_no_match_and_xml_constant_empty(monkeypatch):
    _patch_dive_lookup(monkeypatch, dive_name="dive-alpha")
    monkeypatch.setattr(sut, "LASER_LABELING_CONFIG_XML", "")

    ls = _make_ls(list_result=[])
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    with pytest.raises(Exception) as exc_info:
        await ActivityEnvironment().run(
            sut.create_laser_label_studio_project_activity, 393
        )

    assert "labeling-config XML" in str(exc_info.value)
    ls.projects.create.assert_not_called()


@pytest.mark.asyncio
async def test_falls_back_to_dive_id_when_name_missing(monkeypatch):
    """A dive with a NULL name still gets a unique title from the
    `#{dive_id}` tail alone."""
    _patch_dive_lookup(monkeypatch, dive_name=None)
    expected_title = "#393 - Laser Calibration Labeling"
    monkeypatch.setattr(sut, "LASER_LABELING_CONFIG_XML", "<View/>")

    ls = _make_ls(
        list_result=[],
        create_result=SimpleNamespace(id=202, title=expected_title),
    )
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(
        sut.create_laser_label_studio_project_activity, 393
    )

    assert result == 202
    ls.projects.create.assert_called_once_with(
        title=expected_title,
        label_config="<View/>",
        workspace=None,
    )


@pytest.mark.asyncio
async def test_falls_back_to_dive_id_when_name_too_long(monkeypatch):
    """LS caps `Project.title` at 50 chars. A long dive name is
    truncated to fit, but the `#{dive_id}` tail is always preserved so
    two long-named dives can't collide on a truncated title."""
    long_name = "2024-08-21 Florida Keys reef survey dive 03 (cohort A)"
    assert len(long_name) > sut_utils.LS_PROJECT_TITLE_MAX
    _patch_dive_lookup(monkeypatch, dive_name=long_name)
    expected_title = "2024-08-21 Flori #393 - Laser Calibration Labeling"
    monkeypatch.setattr(sut, "LASER_LABELING_CONFIG_XML", "<View/>")

    ls = _make_ls(
        list_result=[],
        create_result=SimpleNamespace(id=303, title=expected_title),
    )
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(
        sut.create_laser_label_studio_project_activity, 393
    )

    assert result == 303
    assert len(expected_title) <= sut_utils.LS_PROJECT_TITLE_MAX
    ls.projects.create.assert_called_once_with(
        title=expected_title,
        label_config="<View/>",
        workspace=None,
    )


# --------- Garage S3 source-storage registration (ensure_*) ----------


def _make_ls_with_storage(existing_storages=()):
    ls = MagicMock()
    ls.import_storage = MagicMock()
    ls.import_storage.s3 = MagicMock()
    ls.import_storage.s3.list = MagicMock(return_value=list(existing_storages))
    ls.import_storage.s3.create = MagicMock(
        return_value=SimpleNamespace(id=1)
    )
    return ls


@pytest.mark.asyncio
async def test_ensure_s3_storage_registers_presigned_source_when_absent(
    monkeypatch,
):
    """A freshly-created per-dive project gets a Garage S3 source
    storage registered with presign=True so LS can resolve the
    `s3://` task URIs to presigned GET URLs."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    monkeypatch.setenv(
        "E4EFS_OBJECT_STORE__ENDPOINT_URL", "http://garage.example.com"
    )
    monkeypatch.setenv("E4EFS_OBJECT_STORE__ACCESS_KEY", "ak")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__SECRET_KEY", "sk")
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    ls = _make_ls_with_storage(existing_storages=[])
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    async def _run():
        await sut_utils.ensure_label_studio_s3_storage(73)

    await ActivityEnvironment().run(_run)

    ls.import_storage.s3.create.assert_called_once()
    kwargs = ls.import_storage.s3.create.call_args.kwargs
    assert kwargs["project"] == 73
    assert kwargs["bucket"] == "fishsense-test"
    assert kwargs["s3endpoint"] == "http://garage.example.com"
    assert kwargs["presign"] is True
    assert kwargs["use_blob_urls"] is False
    assert kwargs["aws_access_key_id"] == "ak"
    assert kwargs["aws_secret_access_key"] == "sk"


@pytest.mark.asyncio
async def test_ensure_s3_storage_is_idempotent_when_already_registered(
    monkeypatch,
):
    """Re-running create on an existing project must not register a
    duplicate storage — match is on (bucket, title)."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    existing = [
        SimpleNamespace(
            bucket="fishsense-test", title=sut_utils.LS_S3_STORAGE_TITLE
        )
    ]
    ls = _make_ls_with_storage(existing_storages=existing)
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    async def _run():
        await sut_utils.ensure_label_studio_s3_storage(73)

    await ActivityEnvironment().run(_run)

    ls.import_storage.s3.create.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_s3_storage_uses_presign_credentials_when_set(monkeypatch):
    """When a read-only presign key is configured, LS gets THAT key (not
    the main read/write object-store key)."""
    monkeypatch.setenv("E4EFS_OBJECT_STORE__BUCKET", "fishsense-test")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__ACCESS_KEY", "rw-key")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__SECRET_KEY", "rw-secret")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__PRESIGN_ACCESS_KEY", "ro-key")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__PRESIGN_SECRET_KEY", "ro-secret")
    from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel
    cfg.settings.reload()

    ls = _make_ls_with_storage(existing_storages=[])
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    async def _run():
        await sut_utils.ensure_label_studio_s3_storage(73)

    await ActivityEnvironment().run(_run)

    kwargs = ls.import_storage.s3.create.call_args.kwargs
    assert kwargs["aws_access_key_id"] == "ro-key"
    assert kwargs["aws_secret_access_key"] == "ro-secret"

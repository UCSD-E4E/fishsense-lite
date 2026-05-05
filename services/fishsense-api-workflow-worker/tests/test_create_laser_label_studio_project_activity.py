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
    expected_title = "dive-alpha - Laser Calibration Labeling"

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
    expected_title = "dive-alpha - Laser Calibration Labeling"
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
    """A dive with a NULL name still gets a unique title — fall back
    to `f"Dive {dive_id}"` so the create call doesn't generate an
    empty-prefix title that would collide across nameless dives."""
    _patch_dive_lookup(monkeypatch, dive_name=None)
    expected_title = "Dive 393 - Laser Calibration Labeling"
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
    )

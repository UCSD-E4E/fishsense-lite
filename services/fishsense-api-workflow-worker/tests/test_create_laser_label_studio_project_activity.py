"""Unit tests for create_laser_label_studio_project_activity.

Exercises both branches of `create_or_get_label_studio_project`:
title-based idempotent return when a project already exists, and
create-from-XML when it doesn't. The "missing XML" path must raise
rather than create an unlabel-able empty project.

The same `create_or_get_label_studio_project` helper backs all four
stages, so this single test file covers the common code; the
species/headtail/dive_slate variants are thin wrappers and would
just duplicate this coverage.
"""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_returns_existing_project_id_by_title_match(monkeypatch):
    ls = _make_ls(
        list_result=[
            SimpleNamespace(id=42, title="Random Other Project"),
            SimpleNamespace(id=73, title=sut.LASER_PROJECT_TITLE),
        ]
    )
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(
        sut.create_laser_label_studio_project_activity
    )

    assert result == 73
    ls.projects.create.assert_not_called()


@pytest.mark.asyncio
async def test_creates_project_when_no_match_and_xml_present(monkeypatch):
    monkeypatch.setattr(sut, "LASER_LABELING_CONFIG_XML", "<View><Image/></View>")

    ls = _make_ls(
        list_result=[],
        create_result=SimpleNamespace(id=101, title=sut.LASER_PROJECT_TITLE),
    )
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(
        sut.create_laser_label_studio_project_activity
    )

    assert result == 101
    ls.projects.create.assert_called_once_with(
        title=sut.LASER_PROJECT_TITLE,
        label_config="<View><Image/></View>",
    )


@pytest.mark.asyncio
async def test_raises_when_no_match_and_xml_constant_empty(monkeypatch):
    monkeypatch.setattr(sut, "LASER_LABELING_CONFIG_XML", "")

    ls = _make_ls(list_result=[])
    monkeypatch.setattr(sut_utils, "_get_ls_client", lambda: ls)

    with pytest.raises(Exception) as exc_info:
        await ActivityEnvironment().run(
            sut.create_laser_label_studio_project_activity
        )

    assert "labeling-config XML" in str(exc_info.value)
    ls.projects.create.assert_not_called()

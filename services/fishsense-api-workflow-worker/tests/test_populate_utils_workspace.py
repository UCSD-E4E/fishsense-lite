"""LS Enterprise workspace scoping for per-dive project creation.

New per-dive projects must land in the tenant's workspace (`FishSense` in
prod), not the service account's personal one. `create_or_get_label_studio_project`
resolves `label_studio.workspace` to its numeric id, scopes the idempotency
lookup to that workspace, and creates there. Unset → default workspace.
"""

from __future__ import annotations

# Tests exercise the internal `_resolve_workspace_id` helper directly.
# pylint: disable=protected-access

from unittest.mock import MagicMock

import pytest

from fishsense_api_workflow_worker import config as cfg
from fishsense_api_workflow_worker.activities import populate_utils as pu


def _workspace(ws_id: int, title: str):
    w = MagicMock()
    w.id = ws_id
    w.title = title
    return w


def _fake_ls(*, workspaces=(), existing_projects=(), created_id=123):
    ls = MagicMock()
    ls.workspaces.list.return_value = list(workspaces)
    ls.projects.list.return_value = list(existing_projects)
    created = MagicMock()
    created.id = created_id
    ls.projects.create.return_value = created
    return ls


async def _noop(*_a, **_k):
    return None


def test_resolve_workspace_id_matches_by_title(monkeypatch):
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__WORKSPACE", "FishSense")
    cfg.settings.reload()
    ls = _fake_ls(workspaces=[_workspace(3, "Other"), _workspace(7, "FishSense")])
    assert pu._resolve_workspace_id(ls) == 7


def test_resolve_workspace_id_none_when_unset(monkeypatch):
    monkeypatch.delenv("E4EFS_LABEL_STUDIO__WORKSPACE", raising=False)
    cfg.settings.reload()
    ls = _fake_ls(workspaces=[_workspace(7, "FishSense")])
    assert pu._resolve_workspace_id(ls) is None
    ls.workspaces.list.assert_not_called()


def test_resolve_workspace_id_raises_when_configured_but_missing(monkeypatch):
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__WORKSPACE", "Nope")
    cfg.settings.reload()
    ls = _fake_ls(workspaces=[_workspace(7, "FishSense")])
    with pytest.raises(RuntimeError, match="workspace 'Nope' not found"):
        pu._resolve_workspace_id(ls)


async def test_create_scopes_and_creates_in_workspace(monkeypatch):
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__WORKSPACE", "FishSense")
    cfg.settings.reload()
    ls = _fake_ls(workspaces=[_workspace(7, "FishSense")], existing_projects=[])
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(pu, "ensure_label_studio_s3_storage", _noop)

    pid = await pu.create_or_get_label_studio_project(
        project_title="2024 dive 3 - Laser Labeling",
        labeling_config_xml="<View/>",
    )

    assert pid == 123
    ls.projects.list.assert_called_once_with(workspaces=[7])
    _, kwargs = ls.projects.create.call_args
    assert kwargs["workspace"] == 7


async def test_create_idempotent_finds_existing_in_workspace(monkeypatch):
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__WORKSPACE", "FishSense")
    cfg.settings.reload()
    existing = MagicMock()
    existing.id = 55
    existing.title = "X - Laser Labeling"
    ls = _fake_ls(workspaces=[_workspace(7, "FishSense")], existing_projects=[existing])
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(pu, "ensure_label_studio_s3_storage", _noop)

    pid = await pu.create_or_get_label_studio_project(
        project_title="X - Laser Labeling", labeling_config_xml="<View/>"
    )

    assert pid == 55
    ls.projects.create.assert_not_called()


async def test_create_without_workspace_uses_default(monkeypatch):
    monkeypatch.delenv("E4EFS_LABEL_STUDIO__WORKSPACE", raising=False)
    cfg.settings.reload()
    ls = _fake_ls(workspaces=[], existing_projects=[])
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(pu, "ensure_label_studio_s3_storage", _noop)

    pid = await pu.create_or_get_label_studio_project(
        project_title="X - Laser Labeling", labeling_config_xml="<View/>"
    )

    assert pid == 123
    ls.projects.list.assert_called_once_with()  # no workspace filter
    _, kwargs = ls.projects.create.call_args
    assert kwargs["workspace"] is None

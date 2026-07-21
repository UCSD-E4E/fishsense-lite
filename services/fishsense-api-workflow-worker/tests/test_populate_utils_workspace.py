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


async def test_create_does_not_publish(monkeypatch):
    # Projects are created as drafts (no is_published on create) and never
    # published from the create path — publishing is deferred to the populate
    # activities once the task set is complete (see publish_label_studio_project).
    monkeypatch.delenv("E4EFS_LABEL_STUDIO__WORKSPACE", raising=False)
    cfg.settings.reload()
    ls = _fake_ls(workspaces=[], existing_projects=[])
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(pu, "ensure_label_studio_s3_storage", _noop)

    await pu.create_or_get_label_studio_project(
        project_title="X - Laser Labeling", labeling_config_xml="<View/>"
    )

    _, kwargs = ls.projects.create.call_args
    assert "is_published" not in kwargs
    ls.projects.update.assert_not_called()


async def test_publish_label_studio_project_sets_is_published(monkeypatch):
    ls = _fake_ls()
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)

    await pu.publish_label_studio_project(55)

    ls.projects.update.assert_called_once_with(id=55, is_published=True)


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


def test_build_image_url_single_bucket_fallback(monkeypatch):
    # No labels_bucket configured -> falls back to `bucket`, no prefix.
    monkeypatch.delenv("E4EFS_OBJECT_STORE__LABELS_BUCKET", raising=False)
    monkeypatch.delenv("E4EFS_OBJECT_STORE__LABELS_PREFIX", raising=False)
    cfg.settings.reload()
    assert (
        pu.build_image_url("preprocess_jpeg", "abc")
        == "s3://fishsense-test/preprocess_jpeg/abc.JPG"
    )


def test_build_image_url_uses_labels_bucket_and_prefix(monkeypatch):
    monkeypatch.setenv("E4EFS_OBJECT_STORE__LABELS_BUCKET", "labels-fishsense-lite")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__LABELS_PREFIX", "fishsense-lite")
    cfg.settings.reload()
    assert (
        pu.build_image_url("preprocess_groups_jpeg", "caf")
        == "s3://labels-fishsense-lite/fishsense-lite/preprocess_groups_jpeg/caf.JPG"
    )


async def test_ensure_s3_storage_registers_labels_bucket_and_prefix(monkeypatch):
    monkeypatch.setenv("E4EFS_OBJECT_STORE__LABELS_BUCKET", "labels-fishsense-lite")
    monkeypatch.setenv("E4EFS_OBJECT_STORE__LABELS_PREFIX", "fishsense-lite")
    cfg.settings.reload()
    ls = MagicMock()
    ls.import_storage.s3.list.return_value = []
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)

    await pu.ensure_label_studio_s3_storage(999)

    _, kwargs = ls.import_storage.s3.create.call_args
    assert kwargs["bucket"] == "labels-fishsense-lite"
    assert kwargs["prefix"] == "fishsense-lite"
    assert kwargs["presign"] is True


async def _title_for(monkeypatch, dive_id, name, suffix):
    from contextlib import asynccontextmanager  # pylint: disable=import-outside-toplevel
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    fake = MagicMock()

    async def _get(**_k):
        return SimpleNamespace(id=dive_id, name=name)

    fake.dives.get = _get

    @asynccontextmanager
    async def _client():
        yield fake

    monkeypatch.setattr(pu, "get_fs_client", _client)
    return await pu.build_per_dive_title(dive_id, suffix)


async def test_same_named_dives_get_distinct_titles(monkeypatch):
    # Real prod case: dives 439 and 440 share the (mislabeled) name. The
    # #dive_id tail must keep their LS projects distinct.
    t439 = await _title_for(monkeypatch, 439, "101624_AlligatorDeep_FSL02", "Species Labeling")
    t440 = await _title_for(monkeypatch, 440, "101624_AlligatorDeep_FSL02", "Species Labeling")
    assert t439 != t440
    assert "#439" in t439 and "#440" in t440
    assert len(t439) <= pu.LS_PROJECT_TITLE_MAX


async def test_title_truncates_long_name_but_keeps_id(monkeypatch):
    t = await _title_for(monkeypatch, 393, "x" * 80, "Laser Calibration Labeling")
    assert len(t) <= pu.LS_PROJECT_TITLE_MAX
    assert "#393 - Laser Calibration Labeling" in t


async def test_title_nameless_dive_is_id_and_suffix(monkeypatch):
    t = await _title_for(monkeypatch, 7, None, "Species Labeling")
    assert t == "#7 - Species Labeling"


def test_normalize_image_url_decodes_hosted_ls_resolve_wrapper():
    import base64  # pylint: disable=import-outside-toplevel

    s3 = "s3://labels-fishsense-lite/fishsense-lite/preprocess_groups_jpeg/abc.JPG"
    wrapper = "/tasks/999/resolve/?fileuri=" + base64.b64encode(s3.encode()).decode()
    # Hosted LS lists tasks with the resolve-wrapper; must decode back to s3://
    # so dedup/resolve match the built s3:// URLs (else re-import every run).
    assert pu._normalize_image_url(wrapper) == s3
    assert pu._normalize_image_url(s3) == s3          # raw s3:// passes through
    assert pu._normalize_image_url(None) is None
    assert pu._normalize_image_url("/tasks/1/resolve/?fileuri=!!bad") == (
        "/tasks/1/resolve/?fileuri=!!bad"             # undecodable → returned as-is
    )


# ── Labeling-config self-heal ─────────────────────────────────────────
#
# Editing a `<STAGE>_LABELING_CONFIG_XML` constant used to affect only
# projects created after the deploy: `create_or_get_label_studio_project`
# found the existing project by title and returned its id untouched, so
# every already-created per-dive project kept the config it was born with.
# A taxonomy change (e.g. swapping the Fish Model choices) therefore never
# reached annotators. These pin the converge-on-drift behavior.


def _project(pid: int, title: str, label_config=None):
    p = MagicMock()
    p.id = pid
    p.title = title
    p.label_config = label_config
    return p


_CFG_A = '<View><Choices name="x"><Choice value="Old"/></Choices></View>'
_CFG_B = '<View><Choices name="x"><Choice value="New"/></Choices></View>'


async def test_heal_rewrites_config_when_choices_changed():
    ls = _fake_ls()
    changed = await pu.heal_labeling_config(ls, _project(5, "T", _CFG_A), _CFG_B)

    assert changed is True
    ls.projects.update.assert_called_once_with(id=5, label_config=_CFG_B)


async def test_heal_is_noop_when_only_formatting_differs():
    """The anti-churn guard. LS reformats `label_config` server-side, so a
    raw string compare would report drift forever and re-PATCH every project
    on every hourly run."""
    reformatted = (
        '<View>\n'
        '  <Choices   name="x">\n'
        '    <Choice value="Old"></Choice>\n'
        '  </Choices>\n'
        '</View>\n'
    )
    ls = _fake_ls()
    changed = await pu.heal_labeling_config(ls, _project(5, "T", reformatted), _CFG_A)

    assert changed is False
    ls.projects.update.assert_not_called()


async def test_heal_fetches_detail_when_list_omits_config():
    ls = _fake_ls()
    ls.projects.get.return_value = _project(5, "T", _CFG_A)

    changed = await pu.heal_labeling_config(ls, _project(5, "T", None), _CFG_B)

    ls.projects.get.assert_called_once_with(id=5)
    assert changed is True


async def test_heal_survives_ls_rejecting_the_config():
    """LS refuses a config that would invalidate existing annotations. That
    must not fail the whole populate stage."""
    from label_studio_sdk.core import ApiError  # pylint: disable=import-outside-toplevel

    ls = _fake_ls()
    ls.projects.update.side_effect = ApiError(status_code=400, body="in use")

    changed = await pu.heal_labeling_config(ls, _project(5, "T", _CFG_A), _CFG_B)

    assert changed is False  # swallowed, not raised


async def test_create_or_get_heals_existing_project_config(monkeypatch):
    """End of the wiring: an existing per-dive project converges on the
    current constant instead of keeping its birth config."""
    monkeypatch.delenv("E4EFS_LABEL_STUDIO__WORKSPACE", raising=False)
    cfg.settings.reload()
    existing = _project(55, "X - Species Labeling", _CFG_A)
    ls = _fake_ls(workspaces=[], existing_projects=[existing])
    monkeypatch.setattr(pu, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(pu, "ensure_label_studio_s3_storage", _noop)

    pid = await pu.create_or_get_label_studio_project(
        project_title="X - Species Labeling", labeling_config_xml=_CFG_B
    )

    assert pid == 55
    ls.projects.create.assert_not_called()
    ls.projects.update.assert_called_once_with(id=55, label_config=_CFG_B)


def test_species_xml_carries_the_current_fish_model_set():
    """The Fish Model choices are the thing operators actually edit."""
    from fishsense_api_workflow_worker.activities import (  # pylint: disable=import-outside-toplevel
        create_species_label_studio_project_activity as species_sut,
    )

    xml = species_sut.SPECIES_LABELING_CONFIG_XML
    for fish in (
        "Weasly Fish", "Snook", "Grouper", "Shark",
        "Gray Anthias", "Purple Angel", "Yellow Anthias",
    ):
        assert f'<Choice value="{fish}"/>' in xml, fish
    # Retired model names must be gone, or annotators keep seeing them.
    for gone in ("George", "Purple Ant", "Yellow Ant"):
        assert f'<Choice value="{gone}"/>' not in xml, gone

"""Workspace-wide labeling-config reconcile.

The heal inside `create_or_get_label_studio_project` only runs during
populate, and populate stops dispatching for a dive once it is fully
populated. So it converges projects that are still filling and never touches
finished ones — the stable projects labelers actually work in.

Measured in prod on 2026-07-21 after the Fish Model taxonomy swap: all 11
species projects whose dive was still in `needing-species-population` picked
up the new choices; `082923_FishModels_FSL02 #58 - Species Labeling`
(pid 274353), fully populated and out of cohort, kept the old list. This
activity is what closes that gap, so the out-of-cohort case is the one the
tests care most about.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    reconcile_labeling_configs_activity as sut,
)
from fishsense_api_workflow_worker.activities.create_headtail_label_studio_project_activity import (  # pylint: disable=line-too-long
    HEADTAIL_LABELING_CONFIG_XML,
)
from fishsense_api_workflow_worker.activities.create_species_label_studio_project_activity import (  # pylint: disable=line-too-long
    SPECIES_LABELING_CONFIG_XML,
)


def _project(pid: int, title: str, label_config: str | None):
    p = MagicMock()
    p.id = pid
    p.title = title
    p.label_config = label_config
    return p


def _fake_ls(projects):
    ls = MagicMock()
    ls.workspaces.list.return_value = []
    ls.projects.list.return_value = list(projects)
    return ls


_OLD_SPECIES_XML = '<View><Choices name="x"><Choice value="George"/></Choices></View>'


@pytest.fixture(autouse=True)
def _no_workspace(monkeypatch):
    """Workspace unset -> `projects.list()` with no filter (OSS-style)."""
    monkeypatch.setattr(sut, "_resolve_workspace_id", lambda _ls: None)


async def test_heals_a_project_whose_dive_left_the_populate_cohort(monkeypatch):
    """The 274353 case: fully populated, out of cohort, stale config."""
    stale = _project(
        274353, "082923_FishModels_FSL02 #58 - Species Labeling", _OLD_SPECIES_XML
    )
    ls = _fake_ls([stale])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(sut.reconcile_labeling_configs_activity)

    assert result.healed == 1
    ls.projects.update.assert_called_once_with(
        id=274353, label_config=SPECIES_LABELING_CONFIG_XML
    )


async def test_is_a_no_op_when_every_config_is_current(monkeypatch):
    """Runs hourly — an unchanged pass must write nothing."""
    ls = _fake_ls(
        [
            _project(1, "d #1 - Species Labeling", SPECIES_LABELING_CONFIG_XML),
            _project(2, "d #2 - HeadTail Labeling", HEADTAIL_LABELING_CONFIG_XML),
        ]
    )
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(sut.reconcile_labeling_configs_activity)

    assert (result.scanned, result.healed, result.unchanged) == (2, 0, 2)
    ls.projects.update.assert_not_called()


async def test_routes_each_project_to_its_own_stage_config(monkeypatch):
    """A headtail project must never receive the species config."""
    ls = _fake_ls(
        [
            _project(1, "d #1 - Species Labeling", _OLD_SPECIES_XML),
            _project(2, "d #2 - HeadTail Labeling", _OLD_SPECIES_XML),
        ]
    )
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)

    await ActivityEnvironment().run(sut.reconcile_labeling_configs_activity)

    sent = {c.kwargs["id"]: c.kwargs["label_config"] for c in ls.projects.update.call_args_list}
    assert sent[1] == SPECIES_LABELING_CONFIG_XML
    assert sent[2] == HEADTAIL_LABELING_CONFIG_XML


async def test_leaves_projects_it_does_not_own_alone(monkeypatch):
    """The workspace holds unrelated projects (demos, Coral Gardeners).
    Matching on the per-dive title suffix keeps us off them."""
    ls = _fake_ls(
        [
            _project(9, "Coral Gardeners", _OLD_SPECIES_XML),
            _project(10, "Demo Project: Image Captioning", _OLD_SPECIES_XML),
        ]
    )
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(sut.reconcile_labeling_configs_activity)

    assert (result.scanned, result.unrecognized) == (0, 2)
    ls.projects.update.assert_not_called()


async def test_one_rejected_project_does_not_stop_the_pass(monkeypatch):
    """LS refuses a config that would invalidate existing annotations. That
    must not abort the walk — the remaining projects still reconcile."""
    from label_studio_sdk.core import ApiError  # pylint: disable=import-outside-toplevel

    ls = _fake_ls(
        [
            _project(1, "d #1 - Species Labeling", _OLD_SPECIES_XML),
            _project(2, "d #2 - Species Labeling", _OLD_SPECIES_XML),
        ]
    )
    ls.projects.update.side_effect = [ApiError(status_code=400, body="in use"), None]
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)

    result = await ActivityEnvironment().run(sut.reconcile_labeling_configs_activity)

    assert result.scanned == 2
    assert result.healed == 1  # the second one still went through

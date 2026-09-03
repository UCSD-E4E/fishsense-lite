# pylint: disable=protected-access
"""Applying auto-accept verdicts to Label Studio tasks that already exist.

Populate imports an auto-accepted frame already annotated — but it imports a
task exactly once, so a dive whose tasks already exist gets a verdict in the
database and nothing a labeler sees. That is every dive still being labeled,
which is precisely the population auto-accept is meant to relieve.

Same gap the slate detector hit (#493) and the laser pre-annotations hit after
it, fixed the same way: attach to the task instead of re-importing it.

The safety rule here is stricter than for a pre-annotation, because an
annotation is a *claim that the work is done*. A pre-annotation beside a
labeler's own point is noise; a machine annotation on top of one would silently
compete with finished human work. So this only ever touches a task carrying no
annotation and no draft.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    apply_laser_auto_accept_activity as sut,
)


def _prediction(image_id, *, auto_accept=True, x=2000.0, y=1200.0, color="green"):
    return SimpleNamespace(
        image_id=image_id,
        x=x,
        y=y,
        width=4014,
        height=3016,
        color=color,
        auto_accept=auto_accept,
    )


def _label(image_id, task_id, *, completed=False, superseded=False, project_id=73):
    return SimpleNamespace(
        image_id=image_id,
        label_studio_task_id=task_id,
        label_studio_project_id=project_id,
        completed=completed,
        superseded=superseded,
    )


def _task(task_id, *, annotations=(), drafts=()):
    return SimpleNamespace(id=task_id, annotations=list(annotations), drafts=list(drafts))


def _make_fs(predictions, labels):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_predictions = AsyncMock(return_value=predictions)
    fs.labels.get_laser_labels = AsyncMock(return_value=labels)
    return fs


def _make_ls(tasks):
    ls = MagicMock()
    ls.tasks = MagicMock()
    ls.tasks.list.return_value = list(tasks)
    ls.annotations = MagicMock()
    ls.annotations.create.return_value = None
    return ls


# --- target selection --------------------------------------------------------


def test_auto_accepted_prediction_on_an_open_task_is_a_target():
    targets = sut._select_auto_accept_targets([_prediction(1)], [_label(1, 900)])
    assert targets == {1: (900, 73)}


def test_a_prediction_the_gate_did_not_clear_is_not_a_target():
    """Not auto-accepted is the ordinary case: the frame keeps its
    pre-annotation and waits for a person."""
    assert not sut._select_auto_accept_targets(
        [_prediction(1, auto_accept=False)], [_label(1, 900)]
    )


def test_completed_task_is_not_a_target():
    """A labeler already did this one. Nothing to save, and overwriting
    finished human work is the one outcome this feature must never produce."""
    assert not sut._select_auto_accept_targets(
        [_prediction(1)], [_label(1, 900, completed=True)]
    )


def test_superseded_task_is_not_a_target():
    assert not sut._select_auto_accept_targets(
        [_prediction(1)], [_label(1, 900, superseded=True)]
    )


def test_prediction_without_a_dot_is_not_a_target():
    """The gate should never clear an abstention, but if one arrives there is
    nothing placeable to annotate with."""
    assert not sut._select_auto_accept_targets(
        [_prediction(1, x=None, y=None)], [_label(1, 900)]
    )


# --- attaching ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_attaches_an_annotation_to_an_open_unannotated_task(monkeypatch):
    ls = _make_ls([_task(900)])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    applied = await ActivityEnvironment().run(
        sut.apply_laser_auto_accept_for_dive_activity, 5
    )

    assert applied == 1
    kwargs = ls.annotations.create.call_args.kwargs
    assert kwargs["id"] == 900
    result = kwargs["result"][0]
    assert result["origin"] == "prediction"
    assert result["value"]["keypointlabels"] == ["Green Laser"]


@pytest.mark.asyncio
async def test_a_task_that_already_has_an_annotation_is_left_alone(monkeypatch):
    """Doubles as the idempotency test: the first pass leaves an annotation
    behind, so the second pass sees exactly this state. It is also the guard
    against racing a labeler who submitted between the gate and this run."""
    ls = _make_ls([_task(900, annotations=[{"id": 1}])])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    applied = await ActivityEnvironment().run(
        sut.apply_laser_auto_accept_for_dive_activity, 5
    )

    assert applied == 0
    ls.annotations.create.assert_not_called()


@pytest.mark.asyncio
async def test_a_task_with_a_draft_is_left_alone(monkeypatch):
    """A draft is a labeler mid-work on that exact frame. The DB says the task
    is incomplete and LS says it has no annotation yet, so nothing else here
    would notice — annotating underneath them would discard what they are in
    the middle of doing."""
    ls = _make_ls([_task(900, drafts=[{"id": 7}])])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    applied = await ActivityEnvironment().run(
        sut.apply_laser_auto_accept_for_dive_activity, 5
    )

    assert applied == 0
    ls.annotations.create.assert_not_called()


@pytest.mark.asyncio
async def test_the_whole_dive_gets_one_colour(monkeypatch):
    """Same dive-level majority populate uses, so a backfilled annotation
    cannot disagree with its neighbours about the laser's colour."""
    predictions = [_prediction(i, color="red") for i in range(1, 4)]
    predictions[2].color = None
    labels = [_label(i, 900 + i) for i in range(1, 4)]
    ls = _make_ls([_task(900 + i) for i in range(1, 4)])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(predictions, labels))

    await ActivityEnvironment().run(sut.apply_laser_auto_accept_for_dive_activity, 5)

    labels_used = {
        c.kwargs["result"][0]["value"]["keypointlabels"][0]
        for c in ls.annotations.create.call_args_list
    }
    assert labels_used == {"Red Laser"}


@pytest.mark.asyncio
async def test_nothing_eligible_never_touches_label_studio(monkeypatch):
    monkeypatch.setattr(sut, "_get_ls_client", lambda: pytest.fail("LS not needed"))
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs([], []))

    applied = await ActivityEnvironment().run(
        sut.apply_laser_auto_accept_for_dive_activity, 5
    )
    assert applied == 0


@pytest.mark.asyncio
async def test_a_task_missing_from_label_studio_is_skipped_not_fatal(monkeypatch):
    """The DB can name a task the project no longer holds (deleted project,
    hand-cleaned tasks). Unknown means unknown — annotating blind would 404,
    and failing the activity would wedge the dive for every other frame."""
    ls = _make_ls([])
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    applied = await ActivityEnvironment().run(
        sut.apply_laser_auto_accept_for_dive_activity, 5
    )

    assert applied == 0
    ls.annotations.create.assert_not_called()

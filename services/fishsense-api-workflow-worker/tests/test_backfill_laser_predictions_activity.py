# pylint: disable=protected-access
"""Attaching laser predictions to Label Studio tasks that already exist.

This is what makes re-prediction visible. Populate seeds a task's
pre-annotation once, at import time, and dedupes by URL — so for a dive whose
tasks already exist, writing a new `LaserPrediction` changes the database and
nothing a labeler sees. And "tasks already exist" is *every* dive the
re-prediction cohort selects, since it only picks dives still being labeled.

Same gap the slate detector hit and fixed the same way (#493).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    backfill_laser_predictions_activity as sut,
)
from fishsense_shared import laser_model_version_tag


def _prediction(image_id, x=2000.0, y=1200.0, color="green"):
    return SimpleNamespace(
        image_id=image_id, x=x, y=y, width=4014, height=3016, color=color
    )


def _label(image_id, task_id, *, completed=False, superseded=False, project_id=73):
    return SimpleNamespace(
        image_id=image_id,
        label_studio_task_id=task_id,
        label_studio_project_id=project_id,
        completed=completed,
        superseded=superseded,
    )


# --- target selection --------------------------------------------------------


def test_open_task_with_a_placeable_prediction_is_a_target():
    targets = sut._select_attach_targets([_prediction(1)], [_label(1, 900)])
    assert targets == {1: (900, 73)}


def test_completed_task_is_skipped():
    """A labeler already placed that point; a fresh pre-annotation beside it is
    noise at best, and an invitation to second-guess finished work at worst."""
    assert not sut._select_attach_targets(
        [_prediction(1)], [_label(1, 900, completed=True)]
    )


def test_superseded_task_is_skipped():
    assert not sut._select_attach_targets(
        [_prediction(1)], [_label(1, 900, superseded=True)]
    )


def test_a_non_detection_seeds_nothing():
    """x/y None is either "the model found nothing" or "the dot was outside the
    expected-laser region". Neither is placeable."""
    assert not sut._select_attach_targets(
        [_prediction(1, x=None, y=None)], [_label(1, 900)]
    )


def test_an_image_with_no_task_yet_is_not_a_target():
    """Populate will seed it at import time; this activity is only for tasks
    that already exist."""
    assert not sut._select_attach_targets([_prediction(1)], [])


# --- the activity ------------------------------------------------------------


def _make_ls(existing=()):
    ls = MagicMock()
    ls.predictions.list.return_value = list(existing)
    ls.predictions.create.return_value = None
    return ls


def _make_fs(predictions, labels):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_predictions = AsyncMock(return_value=predictions)
    fs.labels.get_laser_labels = AsyncMock(return_value=labels)
    return fs


@pytest.mark.asyncio
async def test_attaches_to_an_open_task(monkeypatch):
    ls = _make_ls()
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    attached = await ActivityEnvironment().run(
        sut.backfill_laser_predictions_for_dive_activity, 5
    )

    assert attached == 1
    kwargs = ls.predictions.create.call_args.kwargs
    assert kwargs["task"] == 900
    assert kwargs["model_version"] == laser_model_version_tag()


@pytest.mark.asyncio
async def test_is_idempotent_against_the_current_version(monkeypatch):
    """LS allows several predictions per task, so without this check a re-run
    would stack duplicates on every firing."""
    ls = _make_ls(
        existing=[SimpleNamespace(task=900, model_version=laser_model_version_tag())]
    )
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    attached = await ActivityEnvironment().run(
        sut.backfill_laser_predictions_for_dive_activity, 5
    )

    assert attached == 0
    ls.predictions.create.assert_not_called()


@pytest.mark.asyncio
async def test_a_task_seeded_by_an_older_version_is_re_attached(monkeypatch):
    """The reason the tag carries the version at all. Under the old bare
    "laser-detector" constant this task would have looked current and the
    improved prediction would never have reached the labeler."""
    ls = _make_ls(
        existing=[SimpleNamespace(task=900, model_version="laser-detector-v1")]
    )
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(
        sut, "get_fs_client", lambda: _make_fs([_prediction(1)], [_label(1, 900)])
    )

    attached = await ActivityEnvironment().run(
        sut.backfill_laser_predictions_for_dive_activity, 5
    )
    assert attached == 1


@pytest.mark.asyncio
async def test_the_whole_dive_gets_one_colour(monkeypatch):
    """Backfilled tasks must not disagree with their neighbours, so the colour
    is the dive's majority, not each frame's own read."""
    ls = _make_ls()
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    predictions = [
        _prediction(1, color="red"),
        _prediction(2, color="green"),
        _prediction(3, color="green"),
    ]
    labels = [_label(1, 900), _label(2, 901), _label(3, 902)]
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(predictions, labels))

    await ActivityEnvironment().run(sut.backfill_laser_predictions_for_dive_activity, 5)

    used = {
        c.kwargs["result"][0]["value"]["keypointlabels"][0]
        for c in ls.predictions.create.call_args_list
    }
    assert used == {"Green Laser"}


@pytest.mark.asyncio
async def test_nothing_eligible_is_a_cheap_no_op(monkeypatch):
    monkeypatch.setattr(sut, "_get_ls_client", lambda: pytest.fail("LS not needed"))
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs([], []))

    assert (
        await ActivityEnvironment().run(
            sut.backfill_laser_predictions_for_dive_activity, 5
        )
        == 0
    )

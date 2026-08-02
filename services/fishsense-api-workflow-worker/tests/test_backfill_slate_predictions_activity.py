"""Unit tests for backfill_slate_predictions_for_dive_activity.

The activity attaches persisted `SlatePrediction` rows to *existing* dive-slate
Label Studio tasks (the populate seeds pre-annotations only at import time and
runs once per dive, so already-populated dives never receive them). It must:
  * only target seeded predictions with an incomplete, non-superseded LS task,
  * be idempotent (skip tasks already carrying a `slate-detector` prediction),
  * apply the same photo->composite keypoint conversion the populate uses.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.slate_prediction import SlatePrediction
from fishsense_api_workflow_worker.activities import (
    backfill_slate_predictions_activity as sut,
)


def _pred(image_id: int, *, points, confidence: float = 0.9) -> SlatePrediction:
    return SlatePrediction(
        id=image_id * 1000,
        reference_points=points,
        confidence=confidence,
        rejected_reason=None if points else "low_confidence",
        width=4014,
        height=3016,
        created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
        image_id=image_id,
    )


def _slate_label(
    image_id: int,
    *,
    task_id: int | None,
    project_id: int | None = 276394,
    completed: bool = False,
    superseded: bool = False,
) -> DiveSlateLabel:
    return DiveSlateLabel(
        id=image_id * 100,
        label_studio_task_id=task_id,
        label_studio_project_id=project_id,
        image_url=None,
        upside_down=None,
        reference_points=None,
        slate_rectangle=None,
        skipped_points=None,
        updated_at=None,
        completed=completed,
        superseded=superseded,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


# --------------------------- pure target selection ---------------------------


def test_select_attach_targets_filters():
    # pylint: disable=protected-access
    predictions = [
        _pred(1, points=[[10.0, 20.0]]),          # seeded + attachable -> yes
        _pred(2, points=[[10.0, 20.0]]),          # completed task -> no
        _pred(3, points=[[10.0, 20.0]]),          # superseded task -> no
        _pred(4, points=None),                    # declined (no points) -> no
        _pred(5, points=[[10.0, 20.0]]),          # no task_id -> no
        _pred(6, points=[[10.0, 20.0]]),          # no LS task row at all -> no
    ]
    slate_labels = [
        _slate_label(1, task_id=4001),
        _slate_label(2, task_id=4002, completed=True),
        _slate_label(3, task_id=4003, superseded=True),
        _slate_label(4, task_id=4004),
        _slate_label(5, task_id=None),
    ]

    targets = sut._select_attach_targets(predictions, slate_labels)

    assert targets == {1: (4001, 276394)}


def test_select_attach_targets_first_non_superseded_wins():
    # pylint: disable=protected-access
    predictions = [_pred(1, points=[[1.0, 1.0]])]
    slate_labels = [
        _slate_label(1, task_id=9001, superseded=True),
        _slate_label(1, task_id=9002),
    ]
    targets = sut._select_attach_targets(predictions, slate_labels)
    assert targets == {1: (9002, 276394)}


# ------------------------------- activity flow -------------------------------


def _make_fs(predictions, slate_labels):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_slate_predictions = AsyncMock(return_value=predictions)
    fs.labels.get_dive_slate_labels = AsyncMock(return_value=slate_labels)
    return fs


def _make_ls(existing_predictions: List | None = None):
    ls = MagicMock()
    ls.predictions = MagicMock()
    ls.predictions.list = MagicMock(return_value=list(existing_predictions or []))
    ls.predictions.create = MagicMock(return_value=SimpleNamespace(id=1))
    return ls


@pytest.mark.asyncio
async def test_attaches_predictions_to_existing_tasks(monkeypatch):
    predictions = [
        _pred(1, points=[[100.0, 50.0]]),
        _pred(3, points=[[200.0, 60.0]]),
    ]
    slate_labels = [
        _slate_label(1, task_id=4001),
        _slate_label(3, task_id=4003),
    ]
    fs = _make_fs(predictions, slate_labels)
    ls = _make_ls()

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    # Fixed panel aspect -> deterministic panel width (avoid object store / PDF).
    monkeypatch.setattr(sut, "_slate_panel_aspect", AsyncMock(return_value=1.5))

    n = await ActivityEnvironment().run(
        sut.backfill_slate_predictions_for_dive_activity, 65
    )

    assert n == 2
    created_tasks = {c.kwargs["task"] for c in ls.predictions.create.call_args_list}
    assert created_tasks == {4001, 4003}
    # each create carries the slate-detector model version + a keypoint result
    for call in ls.predictions.create.call_args_list:
        assert call.kwargs["model_version"] == sut.SLATE_DETECTOR_MODEL_VERSION
        assert call.kwargs["result"]
        assert call.kwargs["result"][0]["type"] == "keypointlabels"


@pytest.mark.asyncio
async def test_idempotent_skips_tasks_with_existing_slate_prediction(monkeypatch):
    predictions = [
        _pred(1, points=[[100.0, 50.0]]),
        _pred(3, points=[[200.0, 60.0]]),
    ]
    slate_labels = [
        _slate_label(1, task_id=4001),
        _slate_label(3, task_id=4003),
    ]
    fs = _make_fs(predictions, slate_labels)
    # task 4001 already has a slate-detector prediction; a foreign one is ignored
    ls = _make_ls(
        existing_predictions=[
            SimpleNamespace(task=4001, model_version=sut.SLATE_DETECTOR_MODEL_VERSION),
            SimpleNamespace(task=4003, model_version="some-other-model"),
        ]
    )

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(sut, "_slate_panel_aspect", AsyncMock(return_value=1.5))

    n = await ActivityEnvironment().run(
        sut.backfill_slate_predictions_for_dive_activity, 65
    )

    assert n == 1
    created_tasks = {c.kwargs["task"] for c in ls.predictions.create.call_args_list}
    assert created_tasks == {4003}


@pytest.mark.asyncio
async def test_no_seeded_predictions_is_noop(monkeypatch):
    predictions = [_pred(4, points=None)]  # declined only
    slate_labels = [_slate_label(4, task_id=4004)]
    fs = _make_fs(predictions, slate_labels)
    ls = _make_ls()

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_get_ls_client", lambda: ls)
    monkeypatch.setattr(sut, "_slate_panel_aspect", AsyncMock(return_value=1.5))

    n = await ActivityEnvironment().run(
        sut.backfill_slate_predictions_for_dive_activity, 65
    )

    assert n == 0
    ls.predictions.list.assert_not_called()
    ls.predictions.create.assert_not_called()

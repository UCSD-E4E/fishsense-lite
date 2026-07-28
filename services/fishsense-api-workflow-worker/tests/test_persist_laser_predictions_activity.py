# pylint: disable=protected-access
"""Unit tests for persist_laser_predictions_activity."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_shared import LaserPredictionResult
from fishsense_api_workflow_worker.activities import (
    persist_laser_predictions_activity as sut,
)


def _make_fs():
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.put_laser_prediction = AsyncMock(return_value=1)
    return fs


@pytest.mark.asyncio
async def test_persists_each_prediction(monkeypatch):
    fs = _make_fs()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    results = [
        LaserPredictionResult(image_id=11, x=1.0, y=2.0, confidence=0.9),
        LaserPredictionResult(image_id=12, x=None, y=None, confidence=0.1),
    ]
    count = await ActivityEnvironment().run(
        sut.persist_laser_predictions_activity, results
    )

    assert count == 2
    assert fs.labels.put_laser_prediction.await_count == 2
    first = fs.labels.put_laser_prediction.await_args_list[0]
    assert first.args[0] == 11
    assert (first.args[1].x, first.args[1].y, first.args[1].confidence) == (1.0, 2.0, 0.9)


@pytest.mark.asyncio
async def test_accepts_dict_payloads(monkeypatch):
    """Across the Temporal boundary results may arrive as plain dicts."""
    fs = _make_fs()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    count = await ActivityEnvironment().run(
        sut.persist_laser_predictions_activity,
        [{"image_id": 11, "x": 1.0, "y": 2.0, "confidence": 0.9}],
    )

    assert count == 1
    assert fs.labels.put_laser_prediction.await_args.args[0] == 11


@pytest.mark.asyncio
async def test_empty_results_writes_nothing(monkeypatch):
    fs = _make_fs()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    count = await ActivityEnvironment().run(
        sut.persist_laser_predictions_activity, []
    )

    assert count == 0
    fs.labels.put_laser_prediction.assert_not_called()

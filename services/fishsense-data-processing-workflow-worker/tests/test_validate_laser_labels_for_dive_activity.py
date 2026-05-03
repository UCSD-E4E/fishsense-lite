"""Unit tests for validate_laser_labels_for_dive_activity (observe-only).

Pins the activity's contract: returns the count of flagged outliers,
emits structured log lines for each, and never mutates the SDK side
(no `put_laser_label` calls — Phase 1 is observe-only).
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_data_processing_workflow_worker.activities import (
    validate_laser_labels_for_dive_activity as sut,
)


def _label(
    label_id: int,
    image_id: int,
    x: float | None,
    y: float | None,
    *,
    completed: bool = True,
    superseded: bool = False,
) -> LaserLabel:
    return LaserLabel(
        id=label_id,
        label_studio_task_id=10_000 + label_id,
        label_studio_project_id=42,
        x=x,
        y=y,
        label="kp-1",
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _make_fs(labels: List[LaserLabel]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=labels)
    fs.labels.put_laser_label = AsyncMock(
        side_effect=AssertionError(
            "Phase 1 must not write back to fishsense-api"
        )
    )
    return fs


def _colinear_labels(n: int, *, image_id_start: int = 1000) -> List[LaserLabel]:
    """n positives on the line y = 0.4*x + 100 with 1px Gaussian noise."""
    import numpy as np

    rng = np.random.default_rng(0)
    xs = np.linspace(50.0, 1500.0, n)
    ys = 0.4 * xs + 100.0 + rng.normal(0.0, 1.0, size=n)
    return [
        _label(label_id=i + 1, image_id=image_id_start + i, x=float(x), y=float(y))
        for i, (x, y) in enumerate(zip(xs, ys))
    ]


@pytest.mark.asyncio
async def test_returns_zero_when_no_labels(monkeypatch):
    fs = _make_fs(labels=[])
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
    assert result == 0


@pytest.mark.asyncio
async def test_returns_zero_below_minimum_positives(monkeypatch):
    # 4 positives with x/y set, 5 sentinel-null rows. Below MIN_POINTS_FOR_LINE.
    labels = _colinear_labels(4) + [
        _label(label_id=900 + i, image_id=2000 + i, x=None, y=None)
        for i in range(5)
    ]
    fs = _make_fs(labels)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
    assert result == 0


@pytest.mark.asyncio
async def test_clean_dive_flags_no_outliers(monkeypatch):
    fs = _make_fs(labels=_colinear_labels(40))
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
    assert result == 0


@pytest.mark.asyncio
async def test_outlier_label_is_flagged_and_reported(monkeypatch, caplog):
    labels = _colinear_labels(40)
    # Bump label index 5 ~50 px off the line — well above 3σ for σ≈1px.
    labels[5].y = labels[5].y + 50.0  # type: ignore[operator]
    fs = _make_fs(labels)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    with caplog.at_level("INFO"):
        result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)

    assert result >= 1
    # The OUTLIER log line must mention this specific laser_label_id so an
    # operator scanning logs can find the row.
    assert any(
        "OUTLIER" in rec.message and f"laser_label_id={labels[5].id}" in rec.message
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_does_not_call_put_laser_label(monkeypatch):
    """Phase 1 invariant: observe-only. The mocked put_laser_label
    raises AssertionError if invoked, so getting through this test
    proves the activity didn't try to write."""
    labels = _colinear_labels(30)
    labels[2].y = labels[2].y + 80.0  # type: ignore[operator]
    fs = _make_fs(labels)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    await env.run(sut.validate_laser_labels_for_dive_activity, 99)

    fs.labels.put_laser_label.assert_not_called()

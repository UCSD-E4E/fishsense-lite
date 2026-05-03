"""Unit tests for validate_laser_labels_for_dive_activity.

Pins the activity's contract: returns the count of flagged outliers,
emits a structured OUTLIER log line for each, and calls
`put_laser_label` with `superseded=True` once per flagged label.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
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
    # put_laser_label echoes the row id back like the real endpoint
    # (status_code=201, body is the persisted row id).
    fs.labels.put_laser_label = AsyncMock(
        side_effect=lambda image_id, label: label.id or 0
    )
    return fs


def _colinear_labels(n: int, *, image_id_start: int = 1000) -> List[LaserLabel]:
    """n positives on the line y = 0.4*x + 100 with 1px Gaussian noise."""
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
async def test_clean_dive_flags_no_outliers_and_does_not_write(monkeypatch):
    fs = _make_fs(labels=_colinear_labels(40))
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
    assert result == 0
    # No outliers → no writes. Pins that the activity doesn't ever
    # supersede a clean dive's labels.
    fs.labels.put_laser_label.assert_not_called()


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
async def test_supersedes_each_flagged_outlier(monkeypatch):
    """Phase 2 invariant: every flagged outlier gets `superseded=True`
    written back via `put_laser_label`. Two outliers in this fixture so
    the assertion catches both an off-by-one and a "wrote only the
    first" regression."""
    labels = _colinear_labels(40)
    labels[3].y = labels[3].y + 60.0  # type: ignore[operator]
    labels[17].y = labels[17].y - 70.0  # type: ignore[operator]
    fs = _make_fs(labels)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)

    assert result == 2
    # One PUT per flagged outlier, each carrying the matching image_id
    # and a `superseded=True` body.
    assert fs.labels.put_laser_label.await_count == 2
    written_image_ids = {
        call.args[0] for call in fs.labels.put_laser_label.call_args_list
    }
    assert written_image_ids == {labels[3].image_id, labels[17].image_id}
    for call in fs.labels.put_laser_label.call_args_list:
        _, written_label = call.args
        assert written_label.superseded is True


@pytest.mark.asyncio
async def test_supersede_failure_propagates(monkeypatch):
    """If the writeback raises, the activity raises so Temporal retries
    the whole run rather than silently leaving outliers in place."""
    labels = _colinear_labels(30)
    labels[2].y = labels[2].y + 80.0  # type: ignore[operator]
    fs = _make_fs(labels)
    fs.labels.put_laser_label = AsyncMock(
        side_effect=RuntimeError("simulated PUT failure")
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    with pytest.raises(RuntimeError, match="simulated PUT failure"):
        await env.run(sut.validate_laser_labels_for_dive_activity, 99)

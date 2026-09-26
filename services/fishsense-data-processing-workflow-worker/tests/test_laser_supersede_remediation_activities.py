"""The remediation activities: plan one dive, and apply a reviewed plan.

The apply activity is the only thing in this change that writes, and each of
its guards is a way a revival could land that nobody reviewed:

* it RE-PLANS from current state and writes only ids still in that plan —
  a label superseded or edited since the dry run, or an id that was never in
  a plan, is refused, not written;
* it skips ids that are already live, so re-applying does nothing;
* every write is `superseded=False, superseded_reason=remediation`, and every
  id is logged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.superseded_reason import SupersededReason
from fishsense_shared.laser_remediation import RemediationDiveRequest
from temporalio.exceptions import ApplicationError
from temporalio.testing import ActivityEnvironment

from fishsense_data_processing_workflow_worker.activities import (
    laser_supersede_remediation as sut,
)

pytestmark = pytest.mark.asyncio

SLOPE, INTERCEPT = 0.4, 100.0
NORMAL = np.array([-SLOPE, 1.0]) / np.hypot(SLOPE, 1.0)


def _labels(superseded=(), outliers=None, n=60):
    rng = np.random.default_rng(0)
    xs = np.linspace(50.0, 1500.0, n)
    perp = rng.normal(0.0, 1.85, size=n)
    for i, off in (outliers or {}).items():
        perp[i] += off
    pts = np.column_stack([xs, SLOPE * xs + INTERCEPT]) + perp[:, None] * NORMAL
    return [
        LaserLabel(
            id=i + 1,
            label_studio_task_id=10_000 + i,
            label_studio_project_id=42,
            x=float(p[0]),
            y=float(p[1]),
            label="kp-1",
            updated_at=None,
            superseded=i in superseded,
            completed=True,
            label_studio_json=None,
            image_id=1000 + i,
            user_id=None,
        )
        for i, p in enumerate(pts)
    ]


class FakeApi:
    """Mirrors the endpoint: superseded rows only on request; writes persist."""

    def __init__(self, labels):
        self.labels = labels
        self.puts = []
        self.fetch_kwargs = []

    async def get_laser_labels(self, _dive_id, **kwargs):
        self.fetch_kwargs.append(kwargs)
        return [
            label.model_copy()
            for label in self.labels
            if kwargs.get("include_superseded") or label.superseded is False
        ]

    async def put_laser_label(self, image_id, label):
        for stored in self.labels:
            if stored.id == label.id:
                stored.superseded = label.superseded
                stored.superseded_reason = label.superseded_reason
        self.puts.append((image_id, label))
        return label.id

    def fs(self):
        fs = MagicMock()
        fs.__aenter__ = AsyncMock(return_value=fs)
        fs.__aexit__ = AsyncMock(return_value=None)
        fs.labels = MagicMock()
        fs.labels.get_laser_labels = AsyncMock(side_effect=self.get_laser_labels)
        fs.labels.get_dive_slate_labels = AsyncMock(return_value=[])
        fs.labels.put_laser_label = AsyncMock(side_effect=self.put_laser_label)
        return fs


def _run(api, monkeypatch, fn, request):
    monkeypatch.setattr(sut, "get_fs_client", api.fs)
    return ActivityEnvironment().run(fn, request)


async def test_the_plan_reads_the_full_population_and_writes_nothing(monkeypatch):
    api = FakeApi(_labels(superseded={9, 20}, outliers={5: 60.0}))

    row = await _run(
        api, monkeypatch, sut.plan_laser_supersede_remediation_activity,
        RemediationDiveRequest(dive_id=7),
    )

    assert api.fetch_kwargs == [{"include_superseded": True}]
    assert not api.puts
    assert row["revive_ids"] == [10, 21]


async def test_the_plan_honours_exclusions(monkeypatch):
    api = FakeApi(_labels(superseded={9, 20}))

    row = await _run(
        api, monkeypatch, sut.plan_laser_supersede_remediation_activity,
        RemediationDiveRequest(dive_id=7, excluded_label_ids=[10]),
    )

    assert row["revive_ids"] == [21]
    assert row["excluded_kept"] == [10]


async def test_apply_revives_exactly_the_reviewed_ids(monkeypatch):
    api = FakeApi(_labels(superseded={9, 20}))

    written = await _run(
        api, monkeypatch, sut.apply_laser_supersede_remediation_activity,
        RemediationDiveRequest(dive_id=7, revive_ids=[10, 21]),
    )

    assert written == 2
    assert {label.id for _, label in api.puts} == {10, 21}
    for image_id, label in api.puts:
        assert image_id == label.image_id
        assert label.superseded is False
        assert label.superseded_reason == SupersededReason.REMEDIATION
        # Everything else round-trips: the PUT is not a rebuild.
        assert label.x is not None and label.completed is True


async def test_re_applying_writes_nothing(monkeypatch):
    api = FakeApi(_labels(superseded={9, 20}))
    request = RemediationDiveRequest(dive_id=7, revive_ids=[10, 21])
    await _run(api, monkeypatch, sut.apply_laser_supersede_remediation_activity, request)
    api.puts.clear()

    again = await _run(
        api, monkeypatch, sut.apply_laser_supersede_remediation_activity, request
    )

    assert again == 0
    assert not api.puts


async def test_apply_refuses_an_id_the_current_plan_does_not_contain(monkeypatch):
    """Label 6 is a genuine outlier: the fit flags it, so no plan revives it,
    whatever the request says."""
    api = FakeApi(_labels(superseded={5, 9}, outliers={5: 60.0}))

    with pytest.raises(ApplicationError) as err:
        await _run(
            api, monkeypatch, sut.apply_laser_supersede_remediation_activity,
            RemediationDiveRequest(dive_id=7, revive_ids=[6, 10]),
        )

    assert err.value.non_retryable
    assert not api.puts, "a refused request writes nothing, not a partial set"


async def test_apply_refuses_an_excluded_label_even_if_asked(monkeypatch):
    api = FakeApi(_labels(superseded={9}))

    with pytest.raises(ApplicationError):
        await _run(
            api, monkeypatch, sut.apply_laser_supersede_remediation_activity,
            RemediationDiveRequest(dive_id=7, revive_ids=[10], excluded_label_ids=[10]),
        )

    assert not api.puts


async def test_apply_logs_every_id(monkeypatch, caplog):
    api = FakeApi(_labels(superseded={9, 20}))

    with caplog.at_level("INFO"):
        await _run(
            api, monkeypatch, sut.apply_laser_supersede_remediation_activity,
            RemediationDiveRequest(dive_id=7, revive_ids=[10, 21]),
        )

    revived = [r.getMessage() for r in caplog.records if "REVIVED" in r.getMessage()]
    assert any("laser_label_id=10" in m for m in revived)
    assert any("laser_label_id=21" in m for m in revived)

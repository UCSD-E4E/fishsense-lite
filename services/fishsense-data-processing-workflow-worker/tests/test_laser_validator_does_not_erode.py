"""Running the laser validator again must not supersede anything new.

The validator runs hourly on every dive whose laser labelling is complete, and
superseding is a dead letter. It used to fetch only the still-live labels, fit
a line through them and supersede what `flag_outliers` flagged — so each run
re-fitted the previous run's survivors. `flag_outliers` estimates its noise
scale from the rows it is handed; removing the flagged tail narrows that
population, the next estimate is smaller, and more labels cross the cut. Pass
after pass.

Measured in prod: dive 521 lost 15, then 7, then 1 label on three consecutive
hourly runs (2026-09-05) with no label edited in between, and a deterministic
replay of the vendored fit reproduces those exact 23 labels. Across all 272
dives the iterated fit reproduces 10,031 of prod's 14,523 superseded labels.
The vendored fit also under-estimated the noise (MAD over folded residuals,
~0.59 sigma), which made every pass worse — but fixing that alone does not
stop it: fishsense-core 4.1.0 still erodes 91 of the 272 dives when iterated
(dive 223 went 9 then 4). fishsense-core #88 found why: the eroding dives'
line drifts or steps during the dive, so labels one line cannot represent get
peeled off a pass at a time. It documents the contract this file pins: fit and
flag the FULL population every run, superseded labels included, in a stable
order.

What that makes the validator:
* one stateless judgement of the whole dive — the flagged set is re-derived
  from scratch each run, never accumulated from the previous run's survivors;
* it only ever supersedes; a superseded label the fit now keeps is left alone
  (reviving is a reviewed operator decision, not something an hourly job does);
* order-invariant with respect to how the API hands rows back.

It does not make a dive's result immune to *new* labels: one added mid-dive can
move RANSAC to a different line and flag a different set, and those supersedes
are permanent. The validator runs only on dives whose labelling is complete, so
that is rare, and it is the residual risk of this design.
"""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_data_processing_workflow_worker.activities import (
    validate_laser_labels_for_dive_activity as sut,
)

SLOPE, INTERCEPT = 0.4, 100.0
NORMAL = np.array([-SLOPE, 1.0]) / np.hypot(SLOPE, 1.0)

pytestmark = pytest.mark.asyncio


def _label(label_id: int, image_id: int, xy, superseded=False) -> LaserLabel:
    return LaserLabel(
        id=label_id,
        label_studio_task_id=10_000 + label_id,
        label_studio_project_id=42,
        x=float(xy[0]),
        y=float(xy[1]),
        label="kp-1",
        updated_at=None,
        superseded=superseded,
        completed=True,
        label_studio_json=None,
        image_id=image_id,
        user_id=None,
    )


def _prod_like_dive(seed: int, n: int = 100, sigma: float = 1.85, offsets=None):
    """~100 positives on one laser line with prod-like perpendicular noise
    (dive 257/424 median ~1.85 px), plus a handful of genuine mislabels 30+ px
    off. `offsets` adds a per-frame shift along the normal (a drifting line)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(50.0, 1500.0, n)
    perp = rng.normal(0.0, sigma, size=n)
    if offsets is not None:
        perp = perp + offsets
    for i, off in zip(rng.choice(n, size=4, replace=False), (35.0, -42.0, 60.0, -90.0)):
        perp[i] += off
    pts = np.column_stack([xs, SLOPE * xs + INTERCEPT]) + perp[:, None] * NORMAL
    return [_label(i + 1, 1000 + i, p) for i, p in enumerate(pts)]


class FakeApi:
    """The dive's labels as the real endpoint serves them: superseded rows are
    hidden unless `include_superseded`, rows come back in (image_id, id)
    order unless `shuffle` is set, and supersedes persist across runs."""

    def __init__(self, labels, *, shuffle_seed: int | None = None):
        self.labels = labels
        self.shuffle_seed = shuffle_seed
        self.puts: list[int] = []
        self.fetch_kwargs: list[dict] = []

    async def get_laser_labels(self, _dive_id, **kwargs):
        self.fetch_kwargs.append(kwargs)
        rows = [
            label.model_copy()
            for label in self.labels
            if kwargs.get("include_superseded") or label.superseded is False
        ]
        rows.sort(key=lambda label: (label.image_id, label.id))
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(rows)
        return rows

    async def put_laser_label(self, _image_id, label):
        for stored in self.labels:
            if stored.id == label.id:
                stored.superseded = label.superseded
        self.puts.append(label.id)
        return label.id

    def fs(self):
        fs = MagicMock()
        fs.__aenter__ = AsyncMock(return_value=fs)
        fs.__aexit__ = AsyncMock(return_value=None)
        fs.labels = MagicMock()
        fs.labels.get_laser_labels = AsyncMock(side_effect=self.get_laser_labels)
        fs.labels.get_dive_slate_labels = AsyncMock(return_value=[])
        fs.labels.put_laser_label = AsyncMock(side_effect=self.put_laser_label)
        fs.dives = MagicMock()
        fs.dives.put_dive_laser_line = AsyncMock(return_value=1)
        return fs


async def _run(api: FakeApi, monkeypatch) -> int:
    monkeypatch.setattr(sut, "get_fs_client", api.fs)
    return await ActivityEnvironment().run(
        sut.validate_laser_labels_for_dive_activity, 99
    )


@pytest.mark.parametrize("seed", range(5))
async def test_a_second_run_supersedes_nothing_new(seed, monkeypatch):
    api = FakeApi(_prod_like_dive(seed))

    first = await _run(api, monkeypatch)
    after_first = list(api.puts)
    second = await _run(api, monkeypatch)

    assert first >= 4, "the four genuine mislabels must still go"
    assert second == 0
    assert api.puts == after_first


async def test_a_drifting_line_does_not_erode_run_after_run(monkeypatch):
    """Dive 223's shape: the line walks from -6.7 to +4.4 px across the dive
    with 1-2 px of noise within any stretch (fishsense-core #88). One line
    cannot represent the ends, so the first run's judgement is imperfect —
    but it must be the ONLY judgement, not the first of many.

    (A step or an offset block reads as two parallel lines and the reflection
    stand-down takes the dive before any flagging, so it tests nothing here.)
    """
    n = 120
    ramp = np.linspace(-6.7, 4.4, n)
    api = FakeApi(_prod_like_dive(11, n=n, offsets=ramp))

    counts = [await _run(api, monkeypatch) for _ in range(4)]

    assert counts[1:] == [0, 0, 0], f"supersedes per run: {counts}"


async def test_the_guard_does_not_depend_on_the_noise_estimate(monkeypatch):
    """The erosion is structural, not a property of one estimator: ANY
    single-pass judge re-applied to its own survivors keeps finding someone.
    Stand in the crudest one — "the farthest dot from the line is an outlier"
    — and the validator must still settle after one run, because it judges
    the same full population every time."""

    def farthest_one(xy, fit, **_kwargs):
        dist = fit.perpendicular_distance(xy[:, 0], xy[:, 1])
        flags = np.zeros(xy.shape[0], dtype=bool)
        flags[int(np.argmax(dist))] = True
        return flags

    monkeypatch.setattr(sut, "flag_outliers", farthest_one)
    api = FakeApi(_prod_like_dive(0))

    counts = [await _run(api, monkeypatch) for _ in range(3)]

    assert counts == [1, 0, 0], f"supersedes per run: {counts}"


async def test_the_validator_reads_the_full_population(monkeypatch):
    api = FakeApi(_prod_like_dive(0))

    await _run(api, monkeypatch)

    assert api.fetch_kwargs == [{"include_superseded": True}]


@pytest.mark.parametrize("shuffle_seed", range(5))
async def test_the_result_does_not_depend_on_row_order(shuffle_seed, monkeypatch):
    """RANSAC picks point pairs by row index, so the same labels in another
    order can settle on another line — core measured dive 257's flag count
    ranging 41-63 over 50 shuffles. The validator must impose its own order."""
    reference = FakeApi(_prod_like_dive(3))
    await _run(reference, monkeypatch)

    shuffled = FakeApi(_prod_like_dive(3), shuffle_seed=shuffle_seed)
    await _run(shuffled, monkeypatch)

    assert sorted(shuffled.puts) == sorted(reference.puts)


async def test_a_superseded_label_the_fit_keeps_is_not_revived(monkeypatch):
    """Most labels the erosion took will now pass. Reviving them is a data
    decision made through the reviewed remediation tool, never here."""
    labels = _prod_like_dive(0)
    labels[10].superseded = True  # a good label an earlier run eroded away
    api = FakeApi(labels)

    await _run(api, monkeypatch)

    assert labels[10].id not in api.puts
    assert labels[10].superseded is True


async def test_an_already_superseded_outlier_is_not_written_or_counted(monkeypatch):
    labels = _prod_like_dive(0)
    first = FakeApi(labels)
    superseded_first = await _run(first, monkeypatch)

    # The same dive, re-served with those outliers already superseded.
    second = FakeApi(first.labels)
    assert await _run(second, monkeypatch) == 0
    assert superseded_first > 0 and not second.puts


async def test_a_legacy_null_superseded_row_is_never_written(monkeypatch):
    """`laserlabel.superseded` was added with no backfill, so NULL rows can
    exist; every resolver already treats them as not live. Writing True over
    one would be a change of meaning nobody asked for."""
    labels = _prod_like_dive(0)
    wild = next(label for label in labels if abs(
        float(np.dot([label.x, label.y - INTERCEPT], NORMAL))) > 30)
    wild.superseded = None
    api = FakeApi(labels)

    await _run(api, monkeypatch)

    assert wild.id not in api.puts


def test_the_fit_is_fishsense_cores_not_a_vendored_copy():
    """The vendored `line_fit.py` estimated noise from the MAD of *absolute*
    residuals — a folded normal, ~0.59 sigma — so its "3 sigma" cut sat at
    ~1.78 sigma and flagged ~7.6% of clean labels. fishsense-core 4.1.0 owns
    the fit with that fixed (#85), and a second copy is how the two drift."""
    import importlib.util

    import fishsense_core.laser as core_laser

    assert sut.fit_dive_line is core_laser.fit_dive_line
    assert sut.flag_outliers is core_laser.flag_outliers
    assert (
        importlib.util.find_spec(
            "fishsense_data_processing_workflow_worker.laser_label_validation.line_fit"
        )
        is None
    )

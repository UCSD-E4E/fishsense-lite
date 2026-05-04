"""Unit tests for validate_laser_labels_for_dive_activity.

Pins the activity's contract: returns the count of flagged outliers,
emits a structured OUTLIER log line for each, and calls
`put_laser_label` with `superseded=True` once per flagged label.
"""

from __future__ import annotations

import asyncio
import re
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
    written back via `put_laser_label`, and the rest of the row's
    fields round-trip unchanged on the merge. Two outliers in this
    fixture so the assertion catches both an off-by-one and a "wrote
    only the first" regression. Field-preservation guards against a
    future refactor that constructs a fresh `LaserLabel(superseded=True)`
    and clobbers x/y/label/etc on the upsert."""
    labels = _colinear_labels(40)
    labels[3].y = labels[3].y + 60.0  # type: ignore[operator]
    labels[17].y = labels[17].y - 70.0  # type: ignore[operator]
    # Snapshot the pre-mutation field values so the round-trip
    # assertions below check what the *labeler* persisted, not what the
    # activity might accidentally overwrite to.
    expected_by_image_id = {
        labels[3].image_id: (
            labels[3].x,
            labels[3].y,
            labels[3].label,
            labels[3].label_studio_task_id,
            labels[3].label_studio_project_id,
        ),
        labels[17].image_id: (
            labels[17].x,
            labels[17].y,
            labels[17].label,
            labels[17].label_studio_task_id,
            labels[17].label_studio_project_id,
        ),
    }
    fs = _make_fs(labels)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    result = await env.run(sut.validate_laser_labels_for_dive_activity, 99)

    assert result == 2
    assert fs.labels.put_laser_label.await_count == 2
    written_image_ids = {
        call.args[0] for call in fs.labels.put_laser_label.call_args_list
    }
    assert written_image_ids == set(expected_by_image_id)
    for call in fs.labels.put_laser_label.call_args_list:
        image_id, written_label = call.args
        # The PUT body's image_id must agree with the URL — the API
        # endpoint overwrites from the URL but a mismatch here means
        # the activity is mutating the wrong row.
        assert written_label.image_id == image_id
        # The one field the activity is allowed to change.
        assert written_label.superseded is True
        # Every other field must round-trip unchanged so the merge
        # doesn't clobber the label on the way back.
        x, y, label, ls_task_id, ls_project_id = expected_by_image_id[image_id]
        assert written_label.x == x
        assert written_label.y == y
        assert written_label.label == label
        assert written_label.label_studio_task_id == ls_task_id
        assert written_label.label_studio_project_id == ls_project_id


@pytest.mark.asyncio
async def test_rerun_after_supersede_is_a_noop(monkeypatch):
    """Phase 2 idempotency-at-the-dive-level: once an outlier has been
    superseded, a re-run sees it filtered out by `get_laser_labels`
    (server-side `superseded=False` filter) and writes nothing.

    Mocked with a stateful fs that mirrors the API filter so this is a
    real idempotency test, not just a "the activity is pure" claim."""
    labels = _colinear_labels(40)
    labels[5].y = labels[5].y + 50.0  # type: ignore[operator]
    superseded_ids: set[int | None] = set()

    async def fake_get_laser_labels(_dive_id: int):
        return [label for label in labels if label.id not in superseded_ids]

    async def fake_put_laser_label(_image_id: int, label: LaserLabel) -> int:
        if label.superseded:
            superseded_ids.add(label.id)
        return label.id or 0

    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(side_effect=fake_get_laser_labels)
    fs.labels.put_laser_label = AsyncMock(side_effect=fake_put_laser_label)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    first = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
    assert first == 1
    assert labels[5].id in superseded_ids
    puts_after_first = fs.labels.put_laser_label.await_count
    assert puts_after_first == 1

    second = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
    assert second == 0
    # Re-run must not write anything new — the server-side filter
    # makes the now-superseded outlier invisible, so the second run
    # has nothing to flag.
    assert fs.labels.put_laser_label.await_count == puts_after_first


@pytest.mark.asyncio
async def test_inlier_fraction_strictly_improves_after_supersede(
    monkeypatch, caplog
):
    """The "iterative tightening" property in a deterministic shape:
    after supersede, the line fit is computed over a strictly cleaner
    population and `inlier_fraction` strictly improves toward 1.0.

    Note: in principle the `LABEL_NOISE_MAD_FLOOR_PX` of 1.0 floors
    the per-dive outlier threshold at 3px regardless of how clean the
    data is, so on production-clean dives the second run never flags
    *additional* borderline labels — `may flag additional` in the
    docstring is the algorithm's behavior in noisier regimes than our
    prod data, not a deterministic property to assert. What IS
    deterministic and worth pinning here: removing outliers
    monotonically improves the fit-quality metric `inlier_fraction`.
    A regression where supersede doesn't actually take effect (e.g.,
    the activity logs `OUTLIER` but the stateful mock's filter
    doesn't drop the row) would surface as identical fractions across
    runs."""
    labels = _colinear_labels(40)
    labels[3].y = labels[3].y + 50.0  # type: ignore[operator]
    labels[17].y = labels[17].y - 80.0  # type: ignore[operator]
    labels[25].y = labels[25].y + 30.0  # type: ignore[operator]
    superseded_ids: set[int | None] = set()

    async def fake_get_laser_labels(_dive_id: int):
        return [label for label in labels if label.id not in superseded_ids]

    async def fake_put_laser_label(_image_id: int, label: LaserLabel) -> int:
        if label.superseded:
            superseded_ids.add(label.id)
        return label.id or 0

    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(side_effect=fake_get_laser_labels)
    fs.labels.put_laser_label = AsyncMock(side_effect=fake_put_laser_label)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    def _parse_inlier_fraction(records) -> float:
        # Pull the most-recent "line fit:" log line and extract the
        # `(N%)` percentage immediately after the inlier count.
        match = next(
            (
                re.search(r"\((\d+)%\)", rec.message)
                for rec in reversed(records)
                if "line fit:" in rec.message
            ),
            None,
        )
        assert match is not None, "no line-fit log line captured"
        return float(match.group(1)) / 100.0

    env = ActivityEnvironment()

    with caplog.at_level("INFO"):
        first = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
        first_fraction = _parse_inlier_fraction(caplog.records)

    assert first == 3
    caplog.clear()

    with caplog.at_level("INFO"):
        second = await env.run(sut.validate_laser_labels_for_dive_activity, 99)
        second_fraction = _parse_inlier_fraction(caplog.records)

    assert second == 0
    # Strict improvement: the second run's inlier_fraction is higher
    # than the first run's because the outliers are gone from the
    # denominator. 1.0 is the achievable maximum (every remaining
    # point is an inlier).
    assert second_fraction > first_fraction
    assert second_fraction == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_heartbeats_fire_during_slow_get_laser_labels(monkeypatch):
    """A dive with thousands of laser labels can produce a
    `get_laser_labels` response large enough that the streamed read
    exceeds `heartbeat_timeout=1m` even though the SDK's per-attempt
    `httpx` `read` timeout is 10s — httpx applies its read timeout per
    byte-gap, not to the whole download, so a slowly-streamed multi-MB
    body just keeps reading until done. The activity must pump
    heartbeats on a fixed interval independent of the await on the GET
    so `heartbeat_timeout` doesn't fire mid-fetch.

    Test shape: lower the pump interval to 0.05s, mock the GET to
    sleep 0.3s, count `activity.heartbeat` calls. With a working pump
    we get ~6 pump-driven calls plus the explicit before/after calls.
    Without a pump (the prior implementation) only the 2 explicit
    calls fire — assert >= 4 to leave headroom."""
    monkeypatch.setattr(sut, "HEARTBEAT_INTERVAL_SECONDS", 0.05)

    heartbeat_calls = 0

    def count_heartbeat(*_args, **_kwargs):
        nonlocal heartbeat_calls
        heartbeat_calls += 1

    monkeypatch.setattr(sut.activity, "heartbeat", count_heartbeat)

    slow_get_duration = 0.3

    async def slow_get_laser_labels(_dive_id: int):
        await asyncio.sleep(slow_get_duration)
        return []

    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(side_effect=slow_get_laser_labels)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    await env.run(sut.validate_laser_labels_for_dive_activity, 99)

    assert heartbeat_calls >= 4, (
        f"expected >= 4 heartbeat calls during a {slow_get_duration}s GET "
        "with pump interval 0.05s (~6 pump fires + explicit calls); "
        f"got {heartbeat_calls} — pump is not running"
    )


@pytest.mark.asyncio
async def test_supersede_writes_run_concurrently(monkeypatch):
    """Phase 2 PUTs must run concurrently (not sequentially). A dive
    with many flagged outliers was blowing `start_to_close` on
    sequential PUTs through Traefik — the supersede loop has to
    parallelize, capped at `SUPERSEDE_CONCURRENCY` so we don't
    saturate the data-worker's outbound HTTP slots or hammer the API.

    Side_effect is gated on an `asyncio.Event` so the test can
    observe `peak_in_flight`: with sequential it would be 1; with
    unbounded it would be `n_outliers`; with the cap it should equal
    `SUPERSEDE_CONCURRENCY`."""
    # 16 outliers, twice the cap, so we can observe saturation rather
    # than just "more than one in flight."
    n_outliers = 2 * sut.SUPERSEDE_CONCURRENCY
    labels = _colinear_labels(60)
    outlier_idxs = list(range(0, n_outliers * 2, 2))[:n_outliers]
    for idx in outlier_idxs:
        labels[idx].y = labels[idx].y + 50.0  # type: ignore[operator]

    in_flight = 0
    peak_in_flight = 0
    release = asyncio.Event()

    async def gated_put(_image_id: int, label: LaserLabel) -> int:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await release.wait()
        finally:
            in_flight -= 1
        return label.id or 0

    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=labels)
    fs.labels.put_laser_label = AsyncMock(side_effect=gated_put)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    env = ActivityEnvironment()
    activity_task = asyncio.create_task(
        env.run(sut.validate_laser_labels_for_dive_activity, 99)
    )
    # Let the activity get going and saturate the cap. The PUT
    # side_effect blocks on `release`, so this sleep is bounded by
    # how long the line-fit + log-emission takes (sub-second on this
    # fixture size).
    await asyncio.sleep(0.5)
    assert peak_in_flight == sut.SUPERSEDE_CONCURRENCY, (
        f"peak in flight was {peak_in_flight}, "
        f"expected exactly {sut.SUPERSEDE_CONCURRENCY}"
    )
    release.set()
    result = await activity_task

    assert result == n_outliers
    assert fs.labels.put_laser_label.await_count == n_outliers


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

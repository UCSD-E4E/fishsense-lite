"""Unit tests for evaluate_laser_auto_accept_activity.

The gate itself is pure and tested in `test_laser_auto_accept.py`. What this
pins is the activity contract around it: it judges the dive's WHOLE prediction
set (not just the newly-predicted images), it writes the verdict back for every
row it judged, and it never leaves a stale `auto_accept=True` standing.
"""

# These tests exercise `_config_from_settings` directly: it is the whole of the
# settings-to-config contract, and the public activity reaches it only through a
# Temporal activity call that would need a full settings object to drive.
# pylint: disable=protected-access

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.laser_prediction import LaserPrediction
from fishsense_data_processing_workflow_worker.activities import (
    evaluate_laser_auto_accept_activity as sut,
)
from fishsense_data_processing_workflow_worker.laser_label_validation.auto_accept import (  # noqa: E501  pylint: disable=line-too-long
    DEFAULT_CONFIG,
)


def _prediction(
    image_id: int,
    x: float | None,
    y: float | None,
    *,
    auto_accept: bool = False,
    gate_verdict: str | None = None,
) -> LaserPrediction:
    return LaserPrediction(
        id=image_id,
        image_id=image_id,
        x=x,
        y=y,
        confidence=1.0,
        width=4014,
        height=3016,
        auto_accept=auto_accept,
        gate_verdict=gate_verdict,
    )


def _collinear(n: int, *, spacing: float = 6.0, jitter: float = 0.4):
    """`n` predictions on one line — the shape of a healthy dive."""
    rng = np.random.default_rng(0)
    xs = 900.0 + spacing * np.arange(n)
    ys = 0.4 * xs + 700.0 + rng.normal(0.0, jitter, size=n)
    return [_prediction(1000 + i, float(x), float(y)) for i, (x, y) in enumerate(zip(xs, ys))]


def _scattered(n: int, *, start: int = 5000):
    """`n` predictions that agree on nothing — the v1-detector shape."""
    rng = np.random.default_rng(1)
    return [
        _prediction(start + i, float(rng.uniform(200, 3800)), float(rng.uniform(200, 2800)))
        for i in range(n)
    ]


def _make_fs(predictions: List[LaserPrediction]):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.labels = MagicMock()
    fs.labels.get_laser_predictions = AsyncMock(return_value=predictions)
    fs.labels.put_laser_prediction = AsyncMock(
        side_effect=lambda image_id, prediction: prediction.id or 0
    )
    return fs


async def _run(monkeypatch, predictions, config=DEFAULT_CONFIG):
    """Run the activity with an explicit gate config.

    Config is injected rather than read from settings: Dynaconf validates every
    validator on first attribute access, so touching the real settings here
    would mean plumbing placeholders for the whole worker into a test about
    line fitting. `_config_from_settings` is covered directly instead.
    """
    fs = _make_fs(predictions)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)
    monkeypatch.setattr(sut, "_resolve_config", lambda: config)
    env = ActivityEnvironment()
    summary = await env.run(sut.evaluate_laser_auto_accept_activity, 7)
    return summary, fs


def _written(fs) -> dict:
    """image_id -> the prediction as it was written back."""
    return {c.args[0]: c.args[1] for c in fs.labels.put_laser_prediction.call_args_list}


@pytest.mark.asyncio
async def test_no_predictions_writes_nothing(monkeypatch):
    summary, fs = await _run(monkeypatch, [])
    assert summary.n_points == 0
    assert not summary.eligible
    fs.labels.put_laser_prediction.assert_not_awaited()


@pytest.mark.asyncio
async def test_healthy_dive_auto_accepts_its_predictions(monkeypatch):
    summary, fs = await _run(monkeypatch, _collinear(40))
    assert summary.eligible
    assert summary.reason is None
    assert summary.auto_accepted > 0
    written = _written(fs)
    accepted = [p for p in written.values() if p.auto_accept]
    assert len(accepted) == summary.auto_accepted
    assert all(p.gate_verdict == "auto_accepted" for p in accepted)
    assert all(p.line_offset_px is not None for p in accepted)


@pytest.mark.asyncio
async def test_dive_without_consensus_auto_accepts_nothing(monkeypatch):
    """The v1 shape. Every frame must route to a human, and the summary has to
    say *why* — 'nothing was auto-accepted' alone cannot distinguish a refusal
    to engage from a dive whose dots were all rejected individually."""
    summary, fs = await _run(monkeypatch, _collinear(20) + _scattered(20))
    assert not summary.eligible
    assert summary.reason == "weak_consensus"
    assert summary.auto_accepted == 0
    assert all(not p.auto_accept for p in _written(fs).values())


@pytest.mark.asyncio
async def test_off_line_prediction_is_not_auto_accepted(monkeypatch):
    predictions = _collinear(40)
    predictions[13].x = float(predictions[13].x) + 300.0
    summary, fs = await _run(monkeypatch, predictions)
    assert summary.eligible
    written = _written(fs)
    assert not written[predictions[13].image_id].auto_accept
    assert written[predictions[13].image_id].gate_verdict == "off_line"


@pytest.mark.asyncio
async def test_a_dive_that_loses_consensus_clears_its_stale_verdicts(monkeypatch):
    """The safety case. A dive can be re-predicted, and the new predictions can
    disagree where the old ones agreed. Rows carrying `auto_accept=True` from
    the previous run must be actively cleared — leaving them standing would let
    a frame skip review on the strength of a fit that no longer holds."""
    predictions = _collinear(20) + _scattered(20)
    for prediction in predictions:
        prediction.auto_accept = True
        prediction.gate_verdict = "auto_accepted"

    summary, fs = await _run(monkeypatch, predictions)

    assert not summary.eligible
    written = _written(fs)
    assert len(written) == len(predictions)
    assert all(not p.auto_accept for p in written.values())


@pytest.mark.asyncio
async def test_unchanged_verdicts_are_not_rewritten(monkeypatch):
    """A dive is re-judged every time the predict parent runs, and the cohort
    revisits dives. Re-PUTting hundreds of unchanged rows each pass is pure
    load on the API, so only genuine changes are written."""
    predictions = _collinear(40)
    _, fs = await _run(monkeypatch, predictions)
    first = dict(_written(fs))
    assert first  # first pass writes

    settled = list(first.values())
    _, fs_again = await _run(monkeypatch, settled)
    assert not _written(fs_again)


@pytest.mark.asyncio
async def test_abstentions_are_judged_but_never_auto_accepted(monkeypatch):
    """A frame the detector found no dot on still needs a person; it is just
    not a disagreement, so it must not count against the dive's consensus."""
    predictions = _collinear(40) + [_prediction(9999, None, None)]
    summary, fs = await _run(monkeypatch, predictions)
    assert summary.eligible
    assert summary.n_points == 40  # the abstention is not part of the fit
    written = _written(fs)
    assert written[9999].gate_verdict == "no_prediction"
    assert not written[9999].auto_accept


@pytest.mark.asyncio
async def test_summary_verdict_counts_cover_every_prediction(monkeypatch):
    predictions = _collinear(40) + [_prediction(9999, None, None)]
    summary, _ = await _run(monkeypatch, predictions)
    assert sum(summary.verdicts.values()) == len(predictions)


# --- the kill switch ---------------------------------------------------------


@pytest.mark.asyncio
async def test_a_disabled_gate_records_verdicts_but_accepts_nothing(monkeypatch):
    """The dark run. Every verdict and margin is still written — that is the
    whole point, it is how you measure what the gate would do to real dives —
    but nothing is marked auto-acceptable, so no frame skips a human."""
    disabled = replace(DEFAULT_CONFIG, enabled=False, audit_sample_rate=0.0)
    summary, fs = await _run(monkeypatch, _collinear(40), config=disabled)

    assert summary.enabled is False
    assert summary.eligible  # the dive itself was fine
    assert summary.auto_accepted == 0
    written = _written(fs)
    assert len(written) == 40
    assert all(not p.auto_accept for p in written.values())
    # ...and the histogram still says what it would have done.
    assert summary.verdicts["auto_accepted"] == 40
    assert all(p.line_offset_px is not None for p in written.values())


@pytest.mark.asyncio
async def test_summary_auto_accepted_counts_the_flag_not_the_verdict(monkeypatch):
    """These two disagree exactly when the gate is off, and callers want the
    flag: the parent uses it to decide whether to walk the dive's Label Studio
    tasks, and with the gate disabled there is nothing there to do."""
    disabled = replace(DEFAULT_CONFIG, enabled=False, audit_sample_rate=0.0)
    summary, _ = await _run(monkeypatch, _collinear(40), config=disabled)
    assert summary.verdicts["auto_accepted"] == 40
    assert summary.auto_accepted == 0


# --- settings mapping --------------------------------------------------------


def _settings(**overrides):
    values = {
        "enabled": True,
        "min_predictions": 20,
        "min_inlier_fraction": 0.75,
        "max_perpendicular_px": 10.0,
        "max_along_line_z": 4.0,
        "audit_sample_rate": 0.10,
    }
    values.update(overrides)
    return SimpleNamespace(laser_auto_accept=SimpleNamespace(**values))


def test_config_is_read_from_settings():
    assert sut._config_from_settings(_settings()) == DEFAULT_CONFIG


def test_the_switch_is_settable_without_a_deploy():
    """The reason every knob lives in settings: turning the gate off, and
    starting a dark run, must not require shipping an image."""
    assert sut._config_from_settings(_settings(enabled=False)).enabled is False


def test_thresholds_are_settable_and_still_validated():
    config = sut._config_from_settings(_settings(min_inlier_fraction=0.9))
    assert config.min_inlier_fraction == 0.9
    with pytest.raises(ValueError):
        sut._config_from_settings(_settings(min_predictions=2))


def test_settings_values_are_coerced():
    """Dynaconf hands back whatever the TOML or the environment override held,
    and an integer audit rate must not turn the sampling into integer maths."""
    config = sut._config_from_settings(_settings(audit_sample_rate=1))
    assert config.audit_sample_rate == 1.0

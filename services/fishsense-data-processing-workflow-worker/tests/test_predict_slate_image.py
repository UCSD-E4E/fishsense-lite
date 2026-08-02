"""Unit tests for the slate-prediction gating (pure logic, no OpenCV/model).

The estimate -> seed/decline decision is where the safety lives (a bad
pre-annotation can be accepted into the corpus as ground truth), so it is a
pure function tested exhaustively here. The rectify + template-render +
estimate_plane path needs a real .ORF + PDF fixture and is an integration test
(follow-on), like the other data-worker stage ports.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fishsense_data_processing_workflow_worker.activities import (
    predict_slate_image as sut,
)


def _estimate(ecc: float, points):
    return SimpleNamespace(ecc_score=ecc, image_points=points)


@pytest.mark.parametrize(
    "name,family",
    [
        ("V-Slate 3", "v-slate"),
        ("v-slate 1", "v-slate"),
        ("Tic-Tac-Toe 6", "tic-tac-toe"),
        ("H-Slate", "h-slate"),
        ("Something Else", "something else"),
    ],
)
def test_slate_family(name, family):
    assert sut.slate_family(name) == family


def test_gate_rejects_unsupported_family():
    est = _estimate(0.99, [[1.0, 1.0]])
    pts, conf, reason = sut.gate_estimate(est, "H-Slate", 4000, 3000)
    assert pts is None and conf == 0.0 and reason == "unsupported_slate_family"


def test_gate_rejects_no_board():
    pts, conf, reason = sut.gate_estimate(None, "V-Slate 1", 4000, 3000)
    assert pts is None and conf == 0.0 and reason == "no_board"


def test_gate_rejects_low_confidence():
    est = _estimate(0.5, [[10.0, 10.0]])
    pts, conf, reason = sut.gate_estimate(est, "V-Slate 1", 4000, 3000)
    assert pts is None and conf == pytest.approx(0.5) and reason == "low_confidence"


def test_gate_rejects_points_off_canvas():
    est = _estimate(0.95, [[10.0, 10.0], [4200.0, 10.0]])  # 2nd x > width
    pts, _conf, reason = sut.gate_estimate(est, "Tic-Tac-Toe 6", 4000, 3000)
    assert pts is None and reason == "points_off_canvas"


def test_gate_accepts_and_returns_points():
    est = _estimate(0.9, [[10.0, 20.0], [3000.0, 2000.0]])
    pts, conf, reason = sut.gate_estimate(est, "V-Slate 4", 4014, 3016)
    assert reason is None
    assert conf == pytest.approx(0.9)
    assert pts == [[10.0, 20.0], [3000.0, 2000.0]]


def test_gate_confidence_boundary_is_inclusive():
    est = _estimate(sut.DEFAULT_MIN_CONFIDENCE, [[1.0, 1.0]])
    pts, _conf, reason = sut.gate_estimate(est, "V-Slate 1", 4000, 3000)
    assert reason is None and pts == [[1.0, 1.0]]

"""Unit tests for the slate-prediction gating (pure logic, no OpenCV/model).

The estimate -> seed/decline decision is where the safety lives (a bad
pre-annotation can be accepted into the corpus as ground truth), so it is a
pure function tested exhaustively here. The rectify + template-render +
estimate_plane path needs a real .ORF + PDF fixture and is an integration test
(follow-on), like the other data-worker stage ports.
"""

from __future__ import annotations

from types import SimpleNamespace

import threading
import time
from concurrent.futures import ThreadPoolExecutor

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


def test_get_masker_returns_the_masker_to_a_concurrent_caller(monkeypatch):
    # pylint: disable=protected-access
    """The "cache the outcome" flag must not be set before the outcome exists.

    Activities run in a real thread pool, and `BoardMasker.from_pretrained()`
    reaches HuggingFace — seconds, not microseconds. A caller arriving during
    that window used to see `_MASKER_LOADED = True`, read a still-unset
    `_MASKER`, and silently take the classical path. Silently, because the
    degradation warning only fires in the `except` branch and nothing raised.
    So the whole first concurrent batch of frames lost the learned mask, with
    nothing in the logs to say so.
    """
    import fishsense_core.slate as slate_mod

    sentinel = object()
    loading_started = threading.Event()

    def _slow_from_pretrained(*_a, **_k):
        loading_started.set()
        time.sleep(0.2)
        return sentinel

    monkeypatch.setattr(slate_mod.BoardMasker, "from_pretrained", _slow_from_pretrained)
    monkeypatch.setattr(sut, "DEFAULT_SLATE_CHECKPOINT_PATH", "")
    monkeypatch.setattr(sut, "_MASKER", None)
    monkeypatch.setattr(sut, "_MASKER_LOADED", False)

    def _late_caller():
        assert loading_started.wait(timeout=5)
        return sut._get_masker()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(sut._get_masker)
        late = pool.submit(_late_caller)
        assert first.result(timeout=10) is sentinel
        assert late.result(timeout=10) is sentinel, (
            "concurrent caller got the classical fallback while the mask was loading"
        )


def test_get_masker_falls_back_to_none_and_caches(monkeypatch):
    # pylint: disable=protected-access
    """A BoardMasker that can't load (no net / no checkpoint) must degrade to
    the classical path (None), and the failure is cached (no per-frame retry)."""
    import fishsense_core.slate as slate_mod

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise RuntimeError("no network")

    monkeypatch.setattr(slate_mod.BoardMasker, "from_pretrained", _boom)
    monkeypatch.setattr(sut, "DEFAULT_SLATE_CHECKPOINT_PATH", "")
    monkeypatch.setattr(sut, "_MASKER", None)
    monkeypatch.setattr(sut, "_MASKER_LOADED", False)

    assert sut._get_masker() is None
    assert sut._get_masker() is None  # cached
    assert calls["n"] == 1  # loaded once, failure cached


def _reset_masker(monkeypatch, checkpoint=""):
    # pylint: disable=protected-access
    monkeypatch.setattr(sut, "DEFAULT_SLATE_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(sut, "_MASKER", None)
    monkeypatch.setattr(sut, "_MASKER_LOADED", False)


@pytest.mark.parametrize(
    ("cuda_available", "expected"), [(True, "cuda"), (False, "cpu")]
)
def test_preferred_device_follows_cuda_availability(
    monkeypatch, cuda_available, expected
):
    # pylint: disable=protected-access
    """`BoardMasker` defaults to device="cpu" and, unlike `LaserDetector`, does
    NOT auto-select CUDA — so without asking, the mask runs on the CPU even on
    a GPU pod. It did exactly that for this activity's whole life."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    assert sut._preferred_device() == expected


def test_preferred_device_is_cpu_when_torch_is_unusable(monkeypatch):
    # pylint: disable=protected-access
    """A broken CUDA runtime must degrade to CPU, not fail the frame. This
    stage has two independent fallbacks under it and neither should be
    triggered by merely asking what device to use."""
    import torch

    def _boom():
        raise RuntimeError("no CUDA driver")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    assert sut._preferred_device() == "cpu"


def test_masker_is_loaded_onto_the_preferred_device(monkeypatch):
    # pylint: disable=protected-access
    """The move to the GPU queue is cosmetic unless the device is passed
    through — pin that it reaches both load paths."""
    import fishsense_core.slate as slate_mod

    monkeypatch.setattr(sut, "_preferred_device", lambda: "cuda")
    seen = {}

    def _from_pretrained(*_a, device="cpu", **_k):
        seen["pretrained"] = device
        return object()

    def _from_checkpoint(_path, *_a, device="cpu", **_k):
        seen["checkpoint"] = device
        return object()

    monkeypatch.setattr(slate_mod.BoardMasker, "from_pretrained", _from_pretrained)
    monkeypatch.setattr(slate_mod.BoardMasker, "from_checkpoint", _from_checkpoint)

    _reset_masker(monkeypatch)
    assert sut._get_masker() is not None
    assert seen == {"pretrained": "cuda"}

    _reset_masker(monkeypatch, checkpoint=__file__)  # any path that exists
    assert sut._get_masker() is not None
    assert seen["checkpoint"] == "cuda"

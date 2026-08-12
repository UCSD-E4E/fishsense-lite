# pylint: disable=protected-access
"""Unit tests for the model-assisted laser predict activity.

The fishsense-core `LaserDetector` (torch, GPU) and `LinearRawImage` are
imported lazily inside the activity, so these tests mock them at the seams
and never need the `[laser-detector]` extra installed:

  1. `_predict_from_raw` builds a LinearRawImage and calls the detector with
     rectified output + the dive's intrinsics, and returns its prediction.
  2. `_get_detector` loads the checkpoint once and caches it per process.
  3. The activity downloads the raw bytes, runs the prediction off-loop, and
     maps the result to a `LaserPredictionResult` keyed by image_id
     (including the no-detection x/y=None case).
"""

from __future__ import annotations

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_data_processing_workflow_worker.activities import (
    predict_laser_image as sut,
)
from fishsense_data_processing_workflow_worker.workflows.predict_laser_images_workflow import (
    LaserPredictionResult,
    PredictLaserImageInput,
)


@pytest.fixture(autouse=True)
def _reset_detector_cache():
    """Each test starts with a cold module-level detector cache."""
    sut._DETECTOR = None
    yield
    sut._DETECTOR = None


def _install_fake_linear_raw_image(monkeypatch, captured):
    """Inject a fake `fishsense_core.image.linear_raw_image.LinearRawImage`
    so `_predict_from_raw`'s lazy import resolves without fishsense-core 2.3."""
    mod = types.ModuleType("fishsense_core.image.linear_raw_image")

    class _LinearRawImage:  # pylint: disable=too-few-public-methods
        def __init__(self, source, *, bayer_upsample="repeat"):
            captured["source"] = source
            captured["bayer_upsample"] = bayer_upsample
            # (H, W, C) — the activity reads dims from image.data.shape.
            self.data = SimpleNamespace(shape=(3000, 4000, 3))

    mod.LinearRawImage = _LinearRawImage
    monkeypatch.setitem(
        sys.modules, "fishsense_core.image.linear_raw_image", mod
    )


# ----------------------------- _predict_from_raw -----------------------------


def test_predict_from_raw_calls_detector_with_rectified_output(monkeypatch):
    captured: dict = {}
    _install_fake_linear_raw_image(monkeypatch, captured)

    predict_calls: dict = {}

    class _FakeDetector:  # pylint: disable=too-few-public-methods
        def predict(self, image, **kwargs):
            predict_calls["image"] = image
            predict_calls["kwargs"] = kwargs
            return SimpleNamespace(x=12.5, y=34.0, confidence=0.9)

    monkeypatch.setattr(sut, "_get_detector", lambda checkpoint_path: _FakeDetector())

    pred, width, height = sut._predict_from_raw(
        b"rawbytes",
        camera_matrix=[[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.1, -0.2, 0.0, 0.0, 0.0],
        wavelength="red",
        checkpoint_path="/models/run3.pt",
    )

    assert (pred.x, pred.y, pred.confidence) == (12.5, 34.0, 0.9)
    assert (width, height) == (4000, 3000)  # from image.data.shape (H, W)
    # Built a LinearRawImage straight from the raw bytes.
    assert captured["source"] == b"rawbytes"
    # Rectified output in labeling space, with the dive's intrinsics + wavelength.
    kw = predict_calls["kwargs"]
    assert kw["rectify_output"] is True
    assert kw["wavelength"] == "red"
    assert kw["camera_matrix"].shape == (3, 3)
    assert kw["distortion"].shape == (5,)


# ------------------------------- _get_detector -------------------------------


def test_get_detector_loads_once_and_caches(monkeypatch):
    loads: list[str] = []

    def _fake_load(checkpoint_path):
        loads.append(checkpoint_path)
        return SimpleNamespace(name="detector")

    monkeypatch.setattr(sut, "_load_detector", _fake_load)

    first = sut._get_detector("/models/run3.pt")
    second = sut._get_detector("/models/run3.pt")

    assert first is second
    assert loads == ["/models/run3.pt"]  # loaded exactly once


def test_get_detector_loads_once_under_concurrency(monkeypatch):
    """Activities run in a real ThreadPoolExecutor (see `worker.py`'s
    `activity_executor` + `max_concurrent_activities`), so N threads can enter
    the lazy init together on a cold pod. Unguarded, each one sees
    `_DETECTOR is None` and loads its own copy of a GPU checkpoint — N times
    the VRAM, on the machine least able to spare it.

    The second caller is released only once the first is *inside* the load, so
    this pins the overlap rather than hoping for it.
    """
    monkeypatch.setattr(sut, "_DETECTOR", None)
    loads: list[str] = []
    loading_started = threading.Event()

    def _slow_load(checkpoint_path):
        loading_started.set()
        time.sleep(0.2)
        loads.append(checkpoint_path)
        return SimpleNamespace(name="detector")

    monkeypatch.setattr(sut, "_load_detector", _slow_load)

    def _late_caller():
        assert loading_started.wait(timeout=5)
        return sut._get_detector("/models/run3.pt")

    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(sut._get_detector, "/models/run3.pt")
        late = [pool.submit(_late_caller) for _ in range(2)]
        results = [first.result(timeout=10)] + [f.result(timeout=10) for f in late]

    assert loads == ["/models/run3.pt"], f"checkpoint loaded {len(loads)}x, want 1"
    assert all(r is results[0] for r in results)


# --------------------------------- activity ----------------------------------


def _make_payload(**overrides) -> PredictLaserImageInput:
    base = {
        "checksum": "abc123",
        "image_id": 42,
        "camera_matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "wavelength": None,
    }
    base.update(overrides)
    return PredictLaserImageInput(**base)


def _mock_object_store(monkeypatch, raw=b"raw"):
    client = MagicMock()
    client.download_raw = AsyncMock(return_value=raw)
    monkeypatch.setattr(sut, "open_object_store_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_activity_returns_mapped_prediction(monkeypatch):
    client = _mock_object_store(monkeypatch, raw=b"ORFBYTES")

    def _fake_predict(raw_bytes, *_args):
        assert raw_bytes == b"ORFBYTES"
        return SimpleNamespace(x=100.0, y=200.0, confidence=0.77), 4000, 3000

    monkeypatch.setattr(sut, "_predict_from_raw", _fake_predict)

    result = await ActivityEnvironment().run(
        sut.predict_laser_image, _make_payload(image_id=7)
    )

    assert isinstance(result, LaserPredictionResult)
    assert result.image_id == 7
    assert (result.x, result.y, result.confidence) == (100.0, 200.0, 0.77)
    assert (result.width, result.height) == (4000, 3000)
    client.download_raw.assert_awaited_once_with("abc123")


@pytest.mark.asyncio
async def test_activity_handles_non_detection(monkeypatch):
    _mock_object_store(monkeypatch)
    monkeypatch.setattr(
        sut,
        "_predict_from_raw",
        lambda *a, **k: (SimpleNamespace(x=None, y=None, confidence=0.05), 4000, 3000),
    )

    result = await ActivityEnvironment().run(
        sut.predict_laser_image, _make_payload(image_id=9)
    )

    assert result.image_id == 9
    assert result.x is None and result.y is None
    assert result.confidence == 0.05


@pytest.mark.asyncio
async def test_activity_validates_dict_payload(monkeypatch):
    """Temporal hands the activity a plain dict across the boundary."""
    _mock_object_store(monkeypatch)
    monkeypatch.setattr(
        sut,
        "_predict_from_raw",
        lambda *a, **k: (SimpleNamespace(x=1.0, y=2.0, confidence=0.5), 4000, 3000),
    )

    result = await ActivityEnvironment().run(
        sut.predict_laser_image, _make_payload(image_id=11).model_dump()
    )

    assert result.image_id == 11
    assert (result.x, result.y) == (1.0, 2.0)

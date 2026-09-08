"""Pure-logic tests for `predict_from_jpeg`, the head/tail predict kernel.

The model sits behind a `segment(image) -> masks` seam, so the whole
crop/gate/keypoint/lift pipeline is exercised here with a stub — no GPU, no
weights, no network. That seam exists precisely so these paths are reachable.
"""

from __future__ import annotations

import sys
import types

import cv2
import numpy as np
import pytest

from fishsense_data_processing_workflow_worker.headtail_geometry import crop_origin
from fishsense_data_processing_workflow_worker.activities.predict_headtail_image import (
    predict_from_jpeg,
)

FRAME_W, FRAME_H = 4014, 3016
CROP_W, CROP_H = 1800, 1350


def _jpeg(width: int = FRAME_W, height: int = FRAME_H) -> bytes:
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


class _Stub:
    """Returns fixed crop-local masks, and records what it was handed."""

    def __init__(self, masks):
        self._masks = masks
        self.seen_shape = None

    def segment(self, image):
        self.seen_shape = image.shape[:2]
        return self._masks


def _fish_mask(cx: int, cy: int, half_len: int = 200, half_h: int = 60):
    """An ellipse, so the head/tail detector has a real principal axis."""
    m = np.zeros((CROP_H, CROP_W), dtype=np.uint8)
    cv2.ellipse(m, (cx, cy), (half_len, half_h), 0, 0, 360, 1, -1)
    return m


def test_predicts_and_lifts_keypoints_into_frame_coordinates():
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))
    stub = _Stub([mask])

    result = predict_from_jpeg(_jpeg(), [laser], stub, image_id=7)

    assert result.status == "predicted"
    assert stub.seen_shape == (CROP_H, CROP_W), "model must see the crop, not the frame"
    assert (result.crop_x, result.crop_y) == (ox, oy)
    # The ellipse spans +/-200 px about the laser, so the lifted keypoints must
    # bracket it in FRAME coordinates — the check that the lift happened.
    assert result.head_x == pytest.approx(laser[0], abs=260)
    assert result.tail_x == pytest.approx(laser[0], abs=260)
    assert abs(result.head_x - result.tail_x) == pytest.approx(400, abs=40)
    assert result.width == FRAME_W and result.height == FRAME_H


def test_keypoints_are_not_left_in_crop_coordinates():
    """Regression guard for the plausible-looking failure: a missing lift puts
    every keypoint within the crop's own extent, which still looks like a fish.
    """
    laser = (3500.0, 2500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    assert ox > 0 and oy > 0, "fixture must use an off-origin crop to be meaningful"
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))

    result = predict_from_jpeg(_jpeg(), [laser], _Stub([mask]), image_id=7)

    assert result.head_x > CROP_W or result.head_y > CROP_H


def test_laser_on_no_mask_abstains():
    mask = _fish_mask(100, 100, 40, 20)
    result = predict_from_jpeg(_jpeg(), [(2000.0, 1500.0)], _Stub([mask]), image_id=7)
    assert result.status == "laser_off_all_fish"
    assert result.head_x is None


def test_no_masks_abstains_as_no_detections():
    result = predict_from_jpeg(_jpeg(), [(2000.0, 1500.0)], _Stub([]), image_id=7)
    assert result.status == "no_detections"


def test_abstentions_still_carry_the_stage_version():
    """The cohort selects on a version mismatch, so a row without one would be
    re-predicted forever."""
    result = predict_from_jpeg(_jpeg(), [(2000.0, 1500.0)], _Stub([]), image_id=7)
    assert result.predictor_version is not None


def test_gate_picks_the_lasered_fish_not_the_largest():
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    big = _fish_mask(300, 300, 400, 150)
    small = _fish_mask(int(laser[0] - ox), int(laser[1] - oy), 120, 40)

    result = predict_from_jpeg(_jpeg(), [laser], _Stub([big, small]), image_id=7)

    assert result.status == "predicted"
    assert result.mask_area_px == pytest.approx(int(np.count_nonzero(small)), rel=0.01)


def test_records_which_laser_label_chose_the_fish():
    """Provenance: a prediction whose laser is later superseded must be
    selectable as stale.

    The dots are placed 500 px apart so the mask can contain one and not the
    other. Real data is tamer than this — of the 276 images (1.6%) with more
    than one valid laser label, the second dot is a median of 0 px and at most
    15 px from the first, i.e. duplicate labels of the same dot. This exercises
    selection logic that the corpus rarely triggers, which is exactly why it
    needs a unit test rather than a real frame.
    """
    laser_a = (1800.0, 1500.0)
    laser_b = (2300.0, 1500.0)
    ox, oy = crop_origin(*laser_a, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser_b[0] - ox), int(laser_b[1] - oy), 120, 50)
    assert mask[int(laser_a[1] - oy), int(laser_a[0] - ox)] == 0, "fixture: A must miss"
    assert mask[int(laser_b[1] - oy), int(laser_b[0] - ox)] == 1, "fixture: B must hit"

    result = predict_from_jpeg(
        _jpeg(), [laser_a, laser_b], _Stub([mask]), image_id=7, laser_label_ids=[11, 22]
    )

    assert result.status == "predicted"
    assert result.laser_label_id == 22


def test_crop_is_centred_on_the_first_laser_point_only():
    """A second dot outside that window cannot be gated on — and in the corpus
    that never happens, which is why cropping on the first point is safe.

    Measured: of the 276 images with more than one valid laser label, zero have
    a second dot further than 15 px from the first, against a half-window of
    675 px. Pinned so the assumption is visible if the labeling ever changes.
    """
    far = (200.0, 200.0)
    laser = (3800.0, 2800.0)
    ox, oy = crop_origin(far[0], far[1], FRAME_W, FRAME_H, CROP_W, CROP_H)
    assert not (ox <= laser[0] < ox + CROP_W and oy <= laser[1] < oy + CROP_H)

    mask = _fish_mask(CROP_W // 2, CROP_H // 2)
    result = predict_from_jpeg(_jpeg(), [far, laser], _Stub([mask]), image_id=7)

    # The far dot defined the window; the second one was never reachable.
    assert (result.crop_x, result.crop_y) == (ox, oy)


def test_silhouette_ratio_is_recorded():
    laser = (2000.0, 1500.0)
    ox, oy = crop_origin(*laser, FRAME_W, FRAME_H, CROP_W, CROP_H)
    mask = _fish_mask(int(laser[0] - ox), int(laser[1] - oy))

    result = predict_from_jpeg(_jpeg(), [laser], _Stub([mask]), image_id=7)

    assert result.silhouette_ratio is not None
    assert 0.05 < result.silhouette_ratio < 1.0


def test_no_laser_points_abstains():
    result = predict_from_jpeg(_jpeg(), [], _Stub([]), image_id=7)
    assert result.status == "laser_off_all_fish"


def test_undecodable_bytes_abstain_rather_than_raise():
    result = predict_from_jpeg(b"not a jpeg", [(1.0, 1.0)], _Stub([]), image_id=7)
    assert result.status == "decode_failed"


class TestMaskConversion:
    """SAM3 returns torch tensors, and on the GPU worker they live on the
    device. `np.asarray` on a CUDA tensor raises rather than converting, so
    the stage would fail on the only machine it is meant to run on while
    passing every test that stubs the backend.
    """

    def test_detaches_and_moves_a_device_tensor(self):
        from fishsense_data_processing_workflow_worker.activities.predict_headtail_image import (
            _to_numpy,
        )

        class _DeviceTensor:
            """Refuses direct conversion, the way a CUDA tensor does."""

            def __init__(self, data):
                self._data = data
                self.detached = False

            def __array__(self, *args, **kwargs):
                raise TypeError("can't convert cuda:0 device type tensor to numpy")

            def detach(self):
                self.detached = True
                return self

            def cpu(self):
                return np.asarray(self._data)

        tensor = _DeviceTensor([[1, 0], [0, 1]])
        out = _to_numpy(tensor)
        assert tensor.detached
        assert out.tolist() == [[1, 0], [0, 1]]

    def test_passes_a_plain_array_through(self):
        from fishsense_data_processing_workflow_worker.activities.predict_headtail_image import (
            _to_numpy,
        )

        assert _to_numpy(np.zeros((2, 2))).shape == (2, 2)


class TestSam3AdapterAutocast:
    """SAM 3.1's weights are bfloat16, and nothing in `Sam3Processor` sets up
    autocast for you.

    Without a context, fp32 activations meet bf16 weights and every frame dies
    in `vitdet.forward` with `mat1 and mat2 must have the same dtype, but got
    BFloat16 and Float`. That is a plain retryable `RuntimeError`, so in prod
    it looped on all 356 images of dive 94 while holding a GPU (2026-09-08),
    and it is invisible to every other test in this module because they all
    stub the very seam this adapter implements.

    Every upstream example opens with
    `torch.autocast("cuda", dtype=torch.bfloat16).__enter__()`; this pins that
    the adapter does the same around both model calls.
    """

    class _RecordingProcessor:
        """A stand-in with `Sam3Processor`'s REAL signatures.

        The shape matters as much as the autocast assertion. `set_image`
        returns the state, `set_text_prompt(prompt, state)` takes it back and
        *is* the inference call, the result is a dict, and there is no
        `predict()`. An earlier version of this adapter called a `predict()`
        that does not exist and read `.masks` off what is really a dict --
        which no stub with invented signatures could ever have caught.
        """

        def __init__(self, device_type):
            self._device_type = device_type
            self.enabled_at_set_image = None
            self.enabled_at_predict = None
            self.dtype_at_predict = None
            self.image_type = None
            self.state_round_tripped = False

        def _sample(self):
            import torch

            return (
                torch.is_autocast_enabled(self._device_type),
                torch.get_autocast_dtype(self._device_type),
            )

        def set_image(self, image, _state=None):
            self.enabled_at_set_image = self._sample()[0]
            self.image_type = type(image).__name__
            return {"sentinel": object()}

        def set_text_prompt(self, _prompt, state):
            self.enabled_at_predict, self.dtype_at_predict = self._sample()
            self.state_round_tripped = "sentinel" in state
            state["masks"] = []
            return state

    def _adapter(self, processor):
        from fishsense_data_processing_workflow_worker.activities.predict_headtail_image import (  # noqa: E501  pylint: disable=line-too-long
            _Sam3Adapter,
        )

        return _Sam3Adapter(processor)

    def test_inference_runs_under_bfloat16_autocast(self):
        import torch

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        processor = self._RecordingProcessor(device_type)

        assert not self._adapter(processor).segment(
            np.zeros((32, 32, 3), dtype=np.uint8)
        )

        assert processor.enabled_at_set_image is True, (
            "the vision backbone runs inside set_image -- that is where the "
            "dtype mismatch was raised"
        )
        assert processor.enabled_at_predict is True
        assert processor.dtype_at_predict is torch.bfloat16

    def test_state_is_carried_from_set_image_into_the_prompt_call(self):
        """`set_text_prompt` raises without the state `set_image` returned."""
        import torch

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        processor = self._RecordingProcessor(device_type)
        self._adapter(processor).segment(np.zeros((32, 32, 3), dtype=np.uint8))
        assert processor.state_round_tripped is True

    def test_the_model_is_handed_a_pil_image_not_the_raw_array(self):
        """`set_image` reads `image.shape[-2:]` for an ndarray, which is
        (width, 3) on an HWC frame -- and those numbers are what every mask is
        interpolated to. A numpy frame yields three-pixel-wide masks, silently.
        """
        import torch

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        processor = self._RecordingProcessor(device_type)
        self._adapter(processor).segment(np.zeros((32, 32, 3), dtype=np.uint8))
        assert processor.image_type == "Image", processor.image_type

    def test_autocast_does_not_leak_past_the_call(self):
        """Entered as a context manager, not `__enter__()` as the notebooks
        do -- an activity thread is reused for the next image."""
        import torch

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self._adapter(self._RecordingProcessor(device_type)).segment(
            np.zeros((32, 32, 3), dtype=np.uint8)
        )
        assert torch.is_autocast_enabled(device_type) is False


class TestFishialFallbackAdapter:
    """The CPU-only backend, used when no GPU is available.

    All three of these fail *silently* if got wrong -- an empty mask, a worse
    mask, or a crash -- which is why each is pinned rather than assumed.
    """

    def _adapter(self, segmentation):
        from fishsense_data_processing_workflow_worker.activities.predict_headtail_image import (  # noqa: E501  pylint: disable=line-too-long
            _FishialAdapter,
        )

        return _FishialAdapter(segmentation)

    class _Recording:
        def __init__(self, labels):
            self._labels = labels
            self.seen = None

        def inference(self, image):
            self.seen = image
            return self._labels

    def test_splits_the_instance_label_map_into_binary_masks(self):
        """`FishSegmentation.inference` returns one array with a distinct
        non-zero integer per fish, not a list of masks."""
        labels = np.zeros((40, 60), dtype=np.int32)
        labels[5:10, 5:15] = 1
        labels[20:25, 30:50] = 7  # ids are arbitrary, not contiguous

        masks = self._adapter(self._Recording(labels)).segment(
            np.zeros((40, 60, 3), dtype=np.uint8)
        )

        assert len(masks) == 2
        assert sorted(int(np.asarray(m).sum()) for m in masks) == [50, 100]
        assert all(np.asarray(m).dtype == bool for m in masks)

    def test_background_is_not_returned_as_a_mask(self):
        labels = np.zeros((20, 40), dtype=np.int32)
        assert self._adapter(self._Recording(labels)).segment(
            np.zeros((20, 40, 3), dtype=np.uint8)
        ) == []

    def test_the_frame_is_passed_through_as_bgr(self):
        """`inference` expects the frame as `RectifiedImage.data` hands it
        over. Converting to RGB measurably degrades it and raises nothing --
        the opposite of what the SAM3 adapter must do."""
        frame = np.zeros((20, 40, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # blue channel only, in BGR
        recorder = self._Recording(np.zeros((20, 40), dtype=np.int32))

        self._adapter(recorder).segment(frame)

        assert np.array_equal(recorder.seen, frame), "frame was converted"

    def test_a_non_landscape_crop_is_refused_rather_than_silently_empty(self):
        """`FishSegmentation` returns an empty mask, with no error, whenever
        width <= height. The 1800x1350 crop is landscape so this holds today,
        but it holds by arithmetic rather than by contract."""
        recorder = self._Recording(np.ones((40, 20), dtype=np.int32))

        masks = self._adapter(recorder).segment(np.zeros((40, 20, 3), dtype=np.uint8))

        assert masks == []
        assert recorder.seen is None, "the model should not have been called"


class TestSam3RequiresAGpu:
    def test_load_segmenter_fails_non_retryably_without_cuda(self, monkeypatch):
        """SAM 3.1 cannot be *built* without a GPU: `build_sam3_image_model`
        allocates its position-encoding cache on a hardcoded `device="cuda"`.

        Retrying cannot help, and left retryable it loops holding the pod --
        which is what all 356 activities of dive 94 did on 2026-09-08.
        """
        from temporalio.exceptions import ApplicationError

        from fishsense_data_processing_workflow_worker.activities import (
            predict_headtail_image as sut,
        )

        monkeypatch.setattr(sut, "cuda_available", lambda: False)

        with pytest.raises(ApplicationError) as excinfo:
            # pylint: disable-next=protected-access
            sut._load_segmenter("/nonexistent.pt")

        assert excinfo.value.non_retryable is True
        assert excinfo.value.type == "NoGpuForSam3"


class TestFallbackSegmenterIsLoaded:
    """`FishSegmentation()` is constructed unloaded.

    `inference` then raises `ValueError: model has not been loaded -- call
    load_model() first`, which is retryable and uncapped, so the fallback
    would loop until the child's 6h timeout: the exact failure this backend
    exists to remove. The other tests stub `inference`, so only this one can
    see it.
    """

    def test_load_model_is_called_before_the_segmenter_is_published(
        self, monkeypatch
    ):
        from fishsense_data_processing_workflow_worker.activities import (
            predict_headtail_image as sut,
        )

        class _Segmentation:
            def __init__(self):
                self.loaded = False

            def load_model(self):
                self.loaded = True

            def inference(self, _image):
                if not self.loaded:
                    raise ValueError(
                        "inference failed: model has not been loaded — "
                        "call load_model() first"
                    )
                return np.zeros((4, 8), dtype=np.int32)

        built = _Segmentation()
        fake_module = types.SimpleNamespace(FishSegmentation=lambda: built)
        monkeypatch.setitem(sys.modules, "fishsense_core", types.ModuleType("x"))
        monkeypatch.setitem(sys.modules, "fishsense_core.fish", fake_module)
        monkeypatch.setattr(sut, "_FALLBACK_SEGMENTER", None)

        got = sut.get_fallback_segmenter()

        assert got is built
        assert built.loaded is True, "load_model() was never called"
        # And the seam works end to end on it, which is what would have failed.
        assert sut._FishialAdapter(got).segment(  # pylint: disable=protected-access
            np.zeros((4, 8, 3), dtype=np.uint8)
        ) == []


class TestNoGpuLeavesExistingRowsAlone:
    """What a GPU-less worker may and may not overwrite.

    Keyed on *whether a row exists*, not on which tier produced it. Keying on
    the tier invites two mistakes: skipping only an exact fallback match, so a
    GPU-less worker downgrades a SAM 3.1 row the moment
    `HEADTAIL_PREDICTOR_VERSION` is bumped during an outage; and ignoring a
    superseded laser, so a row of the wrong fish is never redrawn.
    """

    def _payload(self, **kw):
        from fishsense_shared.preprocess_contracts import PredictHeadtailImage

        base = {
            "image_id": 1,
            "checksum": "abc",
            "laser_points": [[10.0, 10.0]],
            "laser_label_ids": [5],
        }
        base.update(kw)
        return PredictHeadtailImage(**base)

    def test_defaults_mean_first_prediction(self):
        p = self._payload()
        assert p.has_existing_prediction is False
        assert p.existing_laser_superseded is False

    def test_an_existing_row_with_a_live_laser_is_left_alone(self):
        p = self._payload(has_existing_prediction=True)
        assert p.has_existing_prediction and not p.existing_laser_superseded

    def test_a_superseded_laser_is_worth_redrawing_on_any_backend(self):
        p = self._payload(has_existing_prediction=True, existing_laser_superseded=True)
        assert p.existing_laser_superseded


class TestLabelStudioTagFollowsTheRow:
    """The tag is the backfill's idempotency key, so it must name the tier
    that actually produced the row.

    Tagging a fallback prediction as SAM 3.1 makes the later upgrade look
    already-attached, and the labeler keeps the Mask R-CNN keypoints for good
    -- the one way the upgrade queue could upgrade the database while
    changing nothing anyone sees.
    """

    def test_the_two_tiers_get_different_tags(self):
        from fishsense_shared.headtail_predictor import (
            HEADTAIL_FALLBACK_PREDICTOR_VERSION,
            HEADTAIL_PREDICTOR_VERSION,
            headtail_model_version_tag,
        )

        sam3 = headtail_model_version_tag(HEADTAIL_PREDICTOR_VERSION)
        fallback = headtail_model_version_tag(HEADTAIL_FALLBACK_PREDICTOR_VERSION)

        assert sam3 != fallback
        assert headtail_model_version_tag() == sam3, "default is the current tier"

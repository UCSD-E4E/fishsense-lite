"""Pure-logic tests for `predict_from_jpeg`, the head/tail predict kernel.

The model sits behind a `segment(image) -> masks` seam, so the whole
crop/gate/keypoint/lift pipeline is exercised here with a stub — no GPU, no
weights, no network. That seam exists precisely so these paths are reachable.
"""

from __future__ import annotations

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
        import numpy as np

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
        import numpy as np

        from fishsense_data_processing_workflow_worker.activities.predict_headtail_image import (
            _to_numpy,
        )

        assert _to_numpy(np.zeros((2, 2))).shape == (2, 2)

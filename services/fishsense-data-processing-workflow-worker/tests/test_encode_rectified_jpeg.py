"""Pure-logic tests for stage 5.1 rectified-only JPEG encoding.

Stage 5.1's per-image transform is rectify-then-encode with no overlay,
so the only pure-logic step is the JPEG encoding itself. These tests
are minimal — the real verification is the integration + parity tests
against a real .ORF.
"""

import cv2
import numpy as np

from fishsense_data_processing_workflow_worker.activities.preprocess_headtail_image import (
    encode_rectified_jpeg,
)


def test_returns_valid_jpeg_bytes():
    img = np.full((1500, 2000, 3), fill_value=128, dtype=np.uint8)
    out = encode_rectified_jpeg(img)
    assert out[:2] == b"\xff\xd8"
    assert len(out) > 1024


def test_decoded_jpeg_keeps_input_shape():
    img = np.full((1000, 1500, 3), fill_value=64, dtype=np.uint8)
    out = encode_rectified_jpeg(img)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (1000, 1500, 3)


def test_does_not_mutate_input():
    img = np.full((400, 600, 3), fill_value=200, dtype=np.uint8)
    original = img.copy()
    encode_rectified_jpeg(img)
    assert np.array_equal(img, original)

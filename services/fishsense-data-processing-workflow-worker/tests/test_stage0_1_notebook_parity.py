"""Parity test: the worker's stage 0.1 transform must produce
byte-identical output to the original `stage0.1_preprocess_laser_images`
notebook for the same raw input + intrinsics + bbox.

Mirrors the stage-2 parity pattern. Marked `integration` because it
depends on the committed real-`.ORF` fixture.
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage

from fishsense_data_processing_workflow_worker.activities.preprocess_laser_image import (
    _rectify_overlay_bbox_encode,
)


pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]
_BBOX = (1800, 700, 2400, 1600)


@pytest.fixture
def orf_path() -> Path:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE


def _notebook_transform(
    image_path: Path, intrinsics: CameraIntrinsics, bbox: tuple
) -> bytes:
    """Reproduce stage0.1's per-image transform exactly: take a Path,
    build RectifiedImage, draw the green rectangle, write `.JPG` via
    cv2.imwrite."""
    image = RectifiedImage(RawImage(image_path), intrinsics)
    img = image.data
    img = cv2.rectangle(
        img,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[2]), int(bbox[3])),
        (0, 255, 0),
        2,
    )
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.JPG"
        cv2.imwrite(out.as_posix(), img)
        return out.read_bytes()


def test_worker_transform_matches_notebook_byte_for_byte(orf_path: Path):
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(_K, dtype=float),
        distortion_coefficients=np.array(_D, dtype=float),
        camera_id=None,
    )

    nb = _notebook_transform(orf_path, intrinsics, _BBOX)
    wk = _rectify_overlay_bbox_encode(
        orf_path.read_bytes(),
        camera_matrix=_K,
        distortion_coefficients=_D,
        bbox=_BBOX,
    )
    assert nb == wk, "stage 0.1 worker and notebook diverge"

"""Parity test: stage 5.1 worker output matches the original notebook.

Notebook performed: `RawImage(image_path).data` -> `cv2.undistort(...)` ->
`cv2.imwrite('.JPG', ...)`. That happens to be exactly what
`RectifiedImage(RawImage(p), intrinsics)` produces, so this test is
also the proof of that equivalence.
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage

from fishsense_data_processing_workflow_worker.activities.preprocess_headtail_image import (
    _rectify_and_encode_jpeg,
)


pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]


@pytest.fixture
def orf_path() -> Path:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE


def _notebook_transform(image_path: Path, intrinsics: CameraIntrinsics) -> bytes:
    """Reproduce stage 5.1's notebook transform: RawImage(path).data ->
    cv2.undistort -> cv2.imwrite('.JPG')."""
    img = RawImage(image_path).data
    img = cv2.undistort(
        img,
        intrinsics.camera_matrix,
        intrinsics.distortion_coefficients,
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

    nb = _notebook_transform(orf_path, intrinsics)
    wk = _rectify_and_encode_jpeg(
        orf_path.read_bytes(),
        camera_matrix=_K,
        distortion_coefficients=_D,
    )
    assert nb == wk, "stage 5.1 worker and notebook diverge"

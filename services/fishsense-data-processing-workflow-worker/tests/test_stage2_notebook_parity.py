"""Parity test: the worker's per-image transform must produce
byte-identical output to the original `stage2_preprocess_dive_images`
notebook for the same raw input + intrinsics + cluster index.

This is the read-only parity check called for in dont-let-me-forget.md
before deleting the notebook. It does not touch the DB, fishsense-api,
or NAS — just the deterministic rectify + overlay + encode pipeline.

Marked `integration` because it depends on the committed real-`.ORF`
fixture (rawpy decoding of synthetic bytes is not feasible).
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage

from fishsense_data_processing_workflow_worker.activities.preprocess_species_image import (  # noqa: E501  pylint: disable=line-too-long
    _rectify_overlay_encode,
)


pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

# Same identity-ish intrinsics shape the integration test uses; the
# notebook fetches real intrinsics from fishsense-api but the parity
# property is over any single intrinsics value, not over the source.
_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]


@pytest.fixture
def orf_path() -> Path:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE


def _notebook_transform(
    image_path: Path,
    intrinsics: CameraIntrinsics,
    cluster_index_zero_based: int,
    cluster_size: int,
) -> bytes:
    """Reproduce stage2_preprocess_dive_images.ipynb cell 10's per-image
    transform exactly: take a Path, build RectifiedImage, putText with
    the notebook's exact parameters, and write a `.JPG` via cv2.imwrite.
    Returns the file's byte contents."""
    image = RectifiedImage(RawImage(image_path), intrinsics)
    img = image.data
    height, width = img.shape[:2]
    text = f"{cluster_index_zero_based + 1}/{cluster_size}"
    org = (width - 350, height - 75)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 5
    color = (0, 0, 255)
    thickness = 10
    lineType = cv2.LINE_AA
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType)

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.JPG"
        cv2.imwrite(out.as_posix(), img)
        return out.read_bytes()


def _worker_transform(
    raw_bytes: bytes,
    cluster_index_one_based: int,
    cluster_size: int,
) -> bytes:
    return _rectify_overlay_encode(
        raw_bytes,
        camera_matrix=_K,
        distortion_coefficients=_D,
        cluster_index=cluster_index_one_based,
        cluster_size=cluster_size,
    )


def test_worker_transform_matches_notebook_byte_for_byte(orf_path: Path):
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(_K, dtype=float),
        distortion_coefficients=np.array(_D, dtype=float),
        camera_id=None,
    )

    nb = _notebook_transform(orf_path, intrinsics, cluster_index_zero_based=0, cluster_size=5)
    wk = _worker_transform(orf_path.read_bytes(), cluster_index_one_based=1, cluster_size=5)
    assert nb == wk, "worker and notebook transforms diverge on the same inputs"


def test_parity_holds_for_a_mid_cluster_index(orf_path: Path):
    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(_K, dtype=float),
        distortion_coefficients=np.array(_D, dtype=float),
        camera_id=None,
    )
    nb = _notebook_transform(orf_path, intrinsics, cluster_index_zero_based=2, cluster_size=5)
    wk = _worker_transform(orf_path.read_bytes(), cluster_index_one_based=3, cluster_size=5)
    assert nb == wk

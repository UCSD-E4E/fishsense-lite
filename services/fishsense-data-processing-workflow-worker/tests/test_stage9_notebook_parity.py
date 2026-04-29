"""Parity test: stage 9 worker output matches the original notebook for
the same .ORF + slate-PDF + reference points + intrinsics.

Mirrors the stage-2/0.1/5.1 parity pattern. Marked `integration` because
it depends on the committed real-`.ORF` fixture; the slate PDF is
synthesized in-test."""

from io import BytesIO
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pymupdf
import pytest

from fishsense_api_sdk.models.camera_intrinsics import CameraIntrinsics
from fishsense_core.image.raw_image import RawImage
from fishsense_core.image.rectified_image import RectifiedImage

from fishsense_data_processing_workflow_worker.activities.preprocess_slate_image import (
    _build_slate_jpeg,
)


pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]
_DPI = 100
_REF_POINTS = [(50.0, 50.0), (200.0, 200.0), (350.0, 100.0)]


def _make_synthetic_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612.0, height=792.0)
    page.draw_rect(
        pymupdf.Rect(150, 150, 450, 600), color=(0, 0, 0), fill=(0, 0, 0)
    )
    buf = BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def orf_path() -> Path:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE


def _notebook_transform(  # pylint: disable=too-many-locals
    image_path: Path,
    pdf_bytes: bytes,
    intrinsics: CameraIntrinsics,
    dpi: int,
    reference_points: list,
) -> bytes:
    """Reproduce stage 9's notebook cell exactly: render PDF -> grayscale ->
    threshold@125 -> BGR; rectify the raw image; concat scaled PDF on
    the left + image on the right; draw red filled circles + numbered
    text at each scaled reference point; cv2.imwrite as `.JPG`."""
    # PDF render path (writes to a temp .pdf because pymupdf in older
    # versions opens by path; pymupdf.open(stream=...) also works but the
    # notebook used a path -> mirror that to be safe for byte parity).
    with tempfile.TemporaryDirectory() as td:
        pdf_path = Path(td) / "slate.pdf"
        pdf_path.write_bytes(pdf_bytes)
        with pymupdf.open(pdf_path) as document:
            page: pymupdf.Page = document.load_page(0)
            pixmap: pymupdf.Pixmap = page.get_pixmap(dpi=dpi)
            arr = np.frombuffer(pixmap.samples, dtype=np.uint8)
            pdf_image = arr.reshape(pixmap.height, pixmap.width, pixmap.n)
            pdf_image = cv2.cvtColor(pdf_image, cv2.COLOR_RGB2GRAY)
            _, pdf_image = cv2.threshold(
                pdf_image, 125, 255, cv2.THRESH_BINARY
            )
            pdf_image = cv2.cvtColor(pdf_image, cv2.COLOR_GRAY2BGR)

        img = RectifiedImage(RawImage(image_path), intrinsics).data

        img_height, img_width = img.shape[:2]
        pdf_height, pdf_width = pdf_image.shape[:2]
        scale_y = float(img_height) / float(pdf_height)
        new_pdf_height = int(pdf_height * scale_y)
        new_pdf_width = int(pdf_width * scale_y)
        pdf_image = cv2.resize(pdf_image, (new_pdf_width, new_pdf_height))

        new_img = np.zeros(
            (img_height, img_width + new_pdf_width, 3), dtype=np.uint8
        )
        new_img[:, :new_pdf_width, :] = pdf_image
        new_img[:, new_pdf_width:, :] = img

        for idx, point in enumerate(reference_points):
            x, y = point
            x *= scale_y
            y *= scale_y
            x = int(x)
            y = int(y)

            cv2.circle(
                new_img, (x, y), radius=25, color=(0, 0, 255), thickness=-1
            )
            text = f"{idx + 1}"
            org = (x + 20, y - 10)
            cv2.putText(
                new_img,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                5,
                (0, 0, 255),
                10,
                cv2.LINE_AA,
            )

        out = Path(td) / "out.JPG"
        cv2.imwrite(out.as_posix(), new_img)
        return out.read_bytes()


def test_worker_transform_matches_notebook_byte_for_byte(orf_path: Path):
    pdf_bytes = _make_synthetic_pdf()

    intrinsics = CameraIntrinsics(
        camera_matrix=np.array(_K, dtype=float),
        distortion_coefficients=np.array(_D, dtype=float),
        camera_id=None,
    )

    nb = _notebook_transform(
        orf_path, pdf_bytes, intrinsics, _DPI, _REF_POINTS
    )
    wk = _build_slate_jpeg(
        raw_bytes=orf_path.read_bytes(),
        pdf_bytes=pdf_bytes,
        camera_matrix=_K,
        distortion_coefficients=_D,
        slate_dpi=_DPI,
        reference_points=_REF_POINTS,
    )
    assert nb == wk, "stage 9 worker and notebook diverge"

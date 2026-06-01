"""End-to-end stage 9 integration test against the local devcontainer
stack (temporal + MinIO object store). Seeds both the raw .ORF and a
synthesized one-page slate PDF into the object store before running the
workflow."""

import os
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.preprocess_slate_image import (
    preprocess_slate_image,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_slate_images_workflow import (
    PreprocessSlateImagesInput,
    PreprocessSlateImagesWorkflow,
)

from ._object_store_itest import BUCKET, make_s3_client, set_object_store_env

pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]
_DPI = 100
_REF_POINTS = [(50.0, 50.0), (200.0, 200.0)]


def _temporal_target() -> str:
    host = os.environ.get("FISHSENSE_TEMPORAL_HOST", "temporal")
    port = os.environ.get("FISHSENSE_TEMPORAL_PORT", "7233")
    return f"{host}:{port}"


def _make_synthetic_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612.0, height=792.0)
    page.draw_rect(
        pymupdf.Rect(150, 150, 450, 600),
        color=(0, 0, 0),
        fill=(0, 0, 0),
    )
    buf = BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def raw_orf_bytes() -> bytes:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE.read_bytes()


@pytest.fixture
def configure_worker_settings(monkeypatch: pytest.MonkeyPatch):
    set_object_store_env(monkeypatch)
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_workflow_processes_one_image_end_to_end(raw_orf_bytes: bytes):
    checksum = f"itest-stage9-{uuid.uuid4().hex}"
    # Use a per-test slate_id so concurrent runs don't collide.
    slate_id = int(uuid.uuid4().int % 1_000_000)

    pdf_bytes = _make_synthetic_pdf()

    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{checksum}.ORF", Body=raw_orf_bytes)
    s3.put_object(
        Bucket=BUCKET, Key=f"slate_pdf/{slate_id}.pdf", Body=pdf_bytes
    )

    client = await Client.connect(_temporal_target())
    task_queue = f"stage9-itest-{uuid.uuid4().hex}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[PreprocessSlateImagesWorkflow],
        activities=[preprocess_slate_image],
    ):
        await client.execute_workflow(
            PreprocessSlateImagesWorkflow.run,
            PreprocessSlateImagesInput(
                dive_id=-1,
                image_checksums=[checksum],
                slate_id=slate_id,
                slate_dpi=_DPI,
                reference_points=_REF_POINTS,
                camera_matrix=_K,
                distortion_coefficients=_D,
            ),
            id=f"stage9-itest-{uuid.uuid4().hex}",
            task_queue=task_queue,
        )

    out = s3.get_object(
        Bucket=BUCKET, Key=f"preprocess_slate_images_jpeg/{checksum}.JPG"
    )
    content = out["Body"].read()
    assert content[:2] == b"\xff\xd8"

    decoded = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    # Composite = scaled PDF on the left + image on the right.
    # Image is ~4000x3000-ish from the .ORF; expect width strictly
    # larger than the image alone.
    assert decoded.shape[1] > 3000

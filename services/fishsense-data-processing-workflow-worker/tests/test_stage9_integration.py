"""End-to-end stage 9 integration test against the local devcontainer
stack (temporal + nginx static_file_server). Seeds both the raw .ORF
and a synthesized one-page slate PDF onto the file-exchange before
running the workflow."""

import os
import uuid
from io import BytesIO
from pathlib import Path

import cv2
import httpx
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

pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]
_DPI = 100
_REF_POINTS = [(50.0, 50.0), (200.0, 200.0)]


def _exchange_url() -> str:
    return os.environ.get(
        "FISHSENSE_STATIC_FILE_SERVER_URL", "http://static_file_server"
    )


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
    monkeypatch.setenv("E4EFS_STATIC_FILE_SERVER__URL", _exchange_url())
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "http://nas.invalid")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "unused")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "unused")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


@pytest.mark.asyncio
async def test_workflow_processes_one_image_end_to_end(
    raw_orf_bytes: bytes, configure_worker_settings
):
    checksum = f"itest-stage9-{uuid.uuid4().hex}"
    # Use a per-test slate_id so concurrent runs don't collide.
    slate_id = int(uuid.uuid4().int % 1_000_000)

    pdf_bytes = _make_synthetic_pdf()

    async with httpx.AsyncClient(
        base_url=_exchange_url(), timeout=httpx.Timeout(60.0)
    ) as http:
        seed_orf = await http.put(
            f"/api/v1/exchange/raw/{checksum}.ORF", content=raw_orf_bytes
        )
        seed_orf.raise_for_status()
        seed_pdf = await http.put(
            f"/api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf",
            content=pdf_bytes,
        )
        seed_pdf.raise_for_status()

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

        out = await http.get(
            f"/api/v1/exchange/preprocess_slate_images_jpeg/{checksum}.JPG"
        )
        out.raise_for_status()
        assert out.content[:2] == b"\xff\xd8"

        decoded = cv2.imdecode(
            np.frombuffer(out.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        # Composite = scaled PDF on the left + image on the right.
        # Image is ~4000x3000-ish from the .ORF; expect width strictly
        # larger than the image alone.
        assert decoded.shape[1] > 3000

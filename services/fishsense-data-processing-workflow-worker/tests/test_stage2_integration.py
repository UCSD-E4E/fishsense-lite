"""End-to-end stage2 integration test against the local devcontainer
stack (temporal + nginx static_file_server).

Skipped automatically until a real `.ORF` fixture is committed at
tests/fixtures/stage2_sample.ORF — rawpy decoding of synthetic bytes is
not feasible, so this test exercises the actual file format.

Mocks: only the upstream api-worker side. We seed the raw `.ORF` onto
the file-exchange directly (the api-worker would do this in production)
and assert the data-worker produces a valid JPEG at the correct path.
The DB is not touched by stage2 and is therefore not asserted here.
"""

import os
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.preprocess_dive_image import (
    preprocess_dive_image,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_dive_images_workflow import (
    PreprocessDiveImagesInput,
    PreprocessDiveImagesWorkflow,
)

pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

# Identity-ish intrinsics with a tiny barrel distortion so cv2.undistort
# is exercised but the output is still recognizably the input image.
_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]


def _exchange_url() -> str:
    return os.environ.get(
        "FISHSENSE_STATIC_FILE_SERVER_URL", "http://static_file_server"
    )


def _temporal_target() -> str:
    host = os.environ.get("FISHSENSE_TEMPORAL_HOST", "temporal")
    port = os.environ.get("FISHSENSE_TEMPORAL_PORT", "7233")
    return f"{host}:{port}"


@pytest.fixture
def raw_orf_bytes() -> bytes:
    if not _ORF_FIXTURE.exists():
        pytest.skip(
            f"missing fixture {_ORF_FIXTURE}; commit a small real .ORF "
            "raw to enable this integration test"
        )
    return _ORF_FIXTURE.read_bytes()


@pytest.fixture
def configure_worker_settings(monkeypatch: pytest.MonkeyPatch):
    """Set the E4EFS_ envvars the worker's dynaconf config eagerly
    validates on first access. Stage 2 only uses `static_file_server.url`;
    the other settings are placeholders so unrelated validators
    (temporal/nas/fishsense_api) don't reject the test process."""
    monkeypatch.setenv("E4EFS_STATIC_FILE_SERVER__URL", _exchange_url())
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "http://nas.invalid")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "unused")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "unused")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_workflow_processes_one_image_end_to_end(raw_orf_bytes: bytes):
    checksum = f"itest-{uuid.uuid4().hex}"

    async with httpx.AsyncClient(
        base_url=_exchange_url(), timeout=httpx.Timeout(60.0)
    ) as http:
        # Stage the raw input as the api-worker would.
        seed = await http.put(
            f"/api/v1/exchange/raw/{checksum}.ORF", content=raw_orf_bytes
        )
        seed.raise_for_status()

        client = await Client.connect(_temporal_target())
        task_queue = f"stage2-itest-{uuid.uuid4().hex}"

        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[PreprocessDiveImagesWorkflow],
            activities=[preprocess_dive_image],
        ):
            await client.execute_workflow(
                PreprocessDiveImagesWorkflow.run,
                PreprocessDiveImagesInput(
                    dive_id=-1,
                    clusters=[[checksum]],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"stage2-itest-{uuid.uuid4().hex}",
                task_queue=task_queue,
            )

        # Output must be GET-able from the file-exchange.
        out = await http.get(
            f"/api/v1/exchange/preprocess_groups_jpeg/{checksum}.JPG"
        )
        out.raise_for_status()
        assert out.content[:2] == b"\xff\xd8", "downloaded output is not a JPEG"

        decoded = cv2.imdecode(
            np.frombuffer(out.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None, "JPEG failed to decode"
        # Notebook-shape sanity: the rectified image keeps roughly the
        # source dimensions. A real Olympus ORF is 4000x3000ish.
        assert decoded.shape[0] >= 1000 and decoded.shape[1] >= 1000

        # Cleanup: there's no DELETE wired into the exchange yet, and
        # the named volume persists across test runs; the per-test uuid
        # checksum keeps runs isolated.

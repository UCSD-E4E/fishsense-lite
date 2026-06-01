"""End-to-end stage 0.1 integration test against the local devcontainer
stack (temporal + MinIO object store).

Mocks: only the upstream api-worker side. We seed the raw `.ORF` into
the Garage/MinIO `raw/{checksum}.ORF` key directly and assert the
data-worker produces a valid JPEG at `preprocess_jpeg/{checksum}.JPG`.
"""

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.preprocess_laser_image import (
    preprocess_laser_image,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (
    PreprocessLaserImagesInput,
    PreprocessLaserImagesWorkflow,
)

from ._object_store_itest import BUCKET, make_s3_client, set_object_store_env

pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]
_BBOX = [1800, 700, 2400, 1600]


def _temporal_target() -> str:
    host = os.environ.get("FISHSENSE_TEMPORAL_HOST", "temporal")
    port = os.environ.get("FISHSENSE_TEMPORAL_PORT", "7233")
    return f"{host}:{port}"


@pytest.fixture
def raw_orf_bytes() -> bytes:
    if not _ORF_FIXTURE.exists():
        pytest.skip(f"missing fixture {_ORF_FIXTURE}")
    return _ORF_FIXTURE.read_bytes()


@pytest.fixture
def configure_worker_settings(monkeypatch: pytest.MonkeyPatch):
    """Point the worker at the local MinIO object store + temporal."""
    set_object_store_env(monkeypatch)
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_workflow_processes_one_image_end_to_end(raw_orf_bytes: bytes):
    checksum = f"itest-stage01-{uuid.uuid4().hex}"

    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{checksum}.ORF", Body=raw_orf_bytes)

    client = await Client.connect(_temporal_target())
    task_queue = f"stage01-itest-{uuid.uuid4().hex}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[PreprocessLaserImagesWorkflow],
        activities=[preprocess_laser_image],
    ):
        await client.execute_workflow(
            PreprocessLaserImagesWorkflow.run,
            PreprocessLaserImagesInput(
                dive_id=-1,
                image_checksums=[checksum],
                camera_matrix=_K,
                distortion_coefficients=_D,
                bbox=_BBOX,
            ),
            id=f"stage01-itest-{uuid.uuid4().hex}",
            task_queue=task_queue,
        )

    # Output lands at the physical JPEG prefix in the object store.
    out = s3.get_object(Bucket=BUCKET, Key=f"preprocess_jpeg/{checksum}.JPG")
    content = out["Body"].read()
    assert content[:2] == b"\xff\xd8"

    decoded = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape[0] >= 1000 and decoded.shape[1] >= 1000

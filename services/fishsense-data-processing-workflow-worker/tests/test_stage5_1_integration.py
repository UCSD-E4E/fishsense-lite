"""End-to-end stage 5.1 integration test against the local devcontainer
stack (temporal + Garage object store)."""

import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker.activities.preprocess_headtail_image import (
    preprocess_headtail_image,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_headtail_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessHeadtailImagesInput,
    PreprocessHeadtailImagesWorkflow,
)

from ._object_store_itest import BUCKET, make_s3_client, set_object_store_env

pytestmark = pytest.mark.integration


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_ORF_FIXTURE = _FIXTURE_DIR / "stage2_sample.ORF"

_K = [[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]]
_D = [-0.05, 0.01, 0.0, 0.0, 0.0]


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
    set_object_store_env(monkeypatch)
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.invalid")
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures("configure_worker_settings")
async def test_workflow_processes_one_image_end_to_end(raw_orf_bytes: bytes):
    checksum = f"itest-stage51-{uuid.uuid4().hex}"

    s3 = make_s3_client()
    s3.put_object(Bucket=BUCKET, Key=f"raw/{checksum}.ORF", Body=raw_orf_bytes)

    client = await Client.connect(_temporal_target())
    task_queue = f"stage51-itest-{uuid.uuid4().hex}"

    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[PreprocessHeadtailImagesWorkflow],
        activities=[preprocess_headtail_image],
    ):
        await client.execute_workflow(
            PreprocessHeadtailImagesWorkflow.run,
            PreprocessHeadtailImagesInput(
                dive_id=-1,
                image_checksums=[checksum],
                camera_matrix=_K,
                distortion_coefficients=_D,
            ),
            id=f"stage51-itest-{uuid.uuid4().hex}",
            task_queue=task_queue,
        )

    out = s3.get_object(
        Bucket=BUCKET, Key=f"preprocess_headtail_jpeg/{checksum}.JPG"
    )
    content = out["Body"].read()
    assert content[:2] == b"\xff\xd8"

    decoded = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded is not None
    assert decoded.shape[0] >= 1000 and decoded.shape[1] >= 1000

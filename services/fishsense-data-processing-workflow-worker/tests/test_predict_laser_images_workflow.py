"""Workflow contract test for PredictLaserImagesWorkflow.

Runs the workflow against an in-process Temporal test server with a stubbed
activity. Asserts the fan-out shape, per-image arg propagation, and that the
workflow returns one prediction per image keyed by image_id.
"""

import uuid
from typing import List

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from fishsense_shared import PredictLaserImage, PredictLaserImagesInput
from fishsense_data_processing_workflow_worker.workflows.predict_laser_images_workflow import (
    LaserPredictionResult,
    PredictLaserImageInput,
    PredictLaserImagesWorkflow,
)

_K = [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
_D = [-0.1, 0.05, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_workflow_fans_out_and_returns_one_prediction_per_image():
    calls: List[PredictLaserImageInput] = []

    @activity.defn(name="predict_laser_image")
    async def stub_predict_laser_image(
        payload: PredictLaserImageInput,
    ) -> LaserPredictionResult:
        calls.append(payload)
        # Echo image_id into the coords so we can prove the mapping.
        return LaserPredictionResult(
            image_id=payload.image_id,
            x=float(payload.image_id),
            y=float(payload.image_id) * 2,
            confidence=0.9,
        )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-predict",
            workflows=[PredictLaserImagesWorkflow],
            activities=[stub_predict_laser_image],
        ):
            results = await env.client.execute_workflow(
                PredictLaserImagesWorkflow.run,
                PredictLaserImagesInput(
                    dive_id=440,
                    images=[
                        PredictLaserImage(image_id=1, checksum="a"),
                        PredictLaserImage(image_id=2, checksum="b"),
                        PredictLaserImage(image_id=3, checksum="c"),
                    ],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                    wavelength="red",
                ),
                id=f"test-predict-{uuid.uuid4()}",
                task_queue="test-predict",
            )

    assert len(calls) == 3
    by_checksum = {c.checksum: c for c in calls}
    assert set(by_checksum) == {"a", "b", "c"}
    for c in calls:
        assert c.camera_matrix == _K
        assert c.distortion_coefficients == _D
        assert c.wavelength == "red"

    # Result carries one prediction per image, keyed by image_id.
    by_id = {r["image_id"] if isinstance(r, dict) else r.image_id: r for r in results}
    assert set(by_id) == {1, 2, 3}


@pytest.mark.asyncio
async def test_workflow_with_no_images_returns_empty_and_calls_nothing():
    calls: List[PredictLaserImageInput] = []

    @activity.defn(name="predict_laser_image")
    async def stub_predict_laser_image(
        payload: PredictLaserImageInput,
    ) -> LaserPredictionResult:
        calls.append(payload)
        return LaserPredictionResult(
            image_id=payload.image_id, x=None, y=None, confidence=0.0
        )

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="test-predict-empty",
            workflows=[PredictLaserImagesWorkflow],
            activities=[stub_predict_laser_image],
        ):
            results = await env.client.execute_workflow(
                PredictLaserImagesWorkflow.run,
                PredictLaserImagesInput(
                    dive_id=440,
                    images=[],
                    camera_matrix=_K,
                    distortion_coefficients=_D,
                ),
                id=f"test-predict-empty-{uuid.uuid4()}",
                task_queue="test-predict-empty",
            )

    assert not calls
    assert not results

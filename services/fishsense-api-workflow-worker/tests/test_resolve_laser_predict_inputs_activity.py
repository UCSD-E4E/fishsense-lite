# pylint: disable=protected-access
"""Unit tests for resolve_laser_predict_inputs_activity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    resolve_laser_predict_inputs_activity as sut,
)


def _image(image_id, checksum):
    return SimpleNamespace(id=image_id, checksum=checksum)


def _make_fs(*, camera_id=1, images=None, predictions=None, labels=None, intrinsics=True):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=SimpleNamespace(id=1, camera_id=camera_id))
    fs.cameras = MagicMock()
    intr = SimpleNamespace(
        camera_matrix=np.eye(3), distortion_coefficients=np.zeros(5)
    ) if intrinsics else None
    fs.cameras.get_intrinsics = AsyncMock(return_value=intr)
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images or [])
    fs.labels = MagicMock()
    fs.labels.get_laser_predictions = AsyncMock(return_value=predictions or [])
    fs.labels.get_laser_labels = AsyncMock(return_value=labels or [])
    return fs


def test_select_filters_predicted_and_completed():
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    predictions = [SimpleNamespace(image_id=1)]
    # image 2 has a completed (live) laser label -> excluded.
    labels = [SimpleNamespace(image_id=2, label_studio_project_id=73, completed=True)]
    needing = sut._select_images_needing_prediction(images, predictions, labels)
    assert [i.id for i in needing] == [3]


def test_seeded_placeholder_does_not_exclude():
    """A populate-seeded placeholder (project_id set but completed=False) is
    NOT a human label — the image still needs a prediction. Regression for
    dive 84 / project 274728, whose 53 unlabeled images carried placeholder
    rows and got 0 predictions."""
    images = [_image(1, "a")]
    labels = [SimpleNamespace(image_id=1, label_studio_project_id=274728, completed=False)]
    needing = sut._select_images_needing_prediction(images, [], labels)
    assert [i.id for i in needing] == [1]


def test_sentinel_label_does_not_exclude():
    images = [_image(1, "a")]
    labels = [
        SimpleNamespace(image_id=1, label_studio_project_id=None, completed=False)
    ]
    needing = sut._select_images_needing_prediction(images, [], labels)
    assert [i.id for i in needing] == [1]


@pytest.mark.asyncio
async def test_activity_builds_input_dto(monkeypatch):
    fs = _make_fs(
        images=[_image(1, "a"), _image(2, "b")],
        predictions=[SimpleNamespace(image_id=1)],
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_laser_predict_inputs_activity, 1
    )

    assert result.dive_id == 1
    assert [img.image_id for img in result.images] == [2]
    assert [img.checksum for img in result.images] == ["b"]
    assert len(result.camera_matrix) == 3
    assert result.wavelength is None


@pytest.mark.asyncio
async def test_activity_raises_without_intrinsics(monkeypatch):
    fs = _make_fs(images=[_image(1, "a")], intrinsics=False)
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    with pytest.raises(ValueError, match="intrinsics"):
        await ActivityEnvironment().run(
            sut.resolve_laser_predict_inputs_activity, 1
        )

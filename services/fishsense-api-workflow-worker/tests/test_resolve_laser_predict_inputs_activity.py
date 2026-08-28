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


def _image(image_id, checksum, *, is_canonical=True):
    # Canonical by default — the resolvers filter on it, mirroring the
    # cohort selectors. Pass False to model a duplicate frame.
    return SimpleNamespace(
        id=image_id, checksum=checksum, is_canonical=is_canonical
    )


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


def _prediction(image_id, version="current"):
    """A LaserPrediction stub. Defaults to the *current* stage version, which
    is what "already predicted" now means."""
    if version == "current":
        version = sut.LASER_PREDICTOR_VERSION
    return SimpleNamespace(image_id=image_id, predictor_version=version)


def test_select_filters_predicted_and_completed():
    images = [_image(1, "a"), _image(2, "b"), _image(3, "c")]
    # image 1 is predicted *at the current version* -> excluded.
    predictions = [_prediction(1)]
    # image 2 has a completed (live) laser label -> excluded.
    labels = [SimpleNamespace(image_id=2, label_studio_project_id=73, completed=True)]
    needing = sut._select_images_needing_prediction(images, predictions, labels)
    assert [i.id for i in needing] == [3]


def test_a_stale_prediction_does_not_exclude():
    """The point of versioning: an old prediction is work, not a reason to
    skip. `put_laser_prediction` upserts on image_id, so re-running overwrites
    rather than duplicating."""
    images = [_image(1, "a"), _image(2, "b")]
    predictions = [_prediction(1, version=sut.LASER_PREDICTOR_VERSION - 1)]
    needing = sut._select_images_needing_prediction(images, predictions, [])
    assert [i.id for i in needing] == [1, 2]


def test_a_prediction_predating_versioning_does_not_exclude():
    """The 1,555 rows already in prod carry NULL, which is 'unknown, therefore
    stale'."""
    images = [_image(1, "a")]
    needing = sut._select_images_needing_prediction(
        images, [_prediction(1, version=None)], []
    )
    assert [i.id for i in needing] == [1]


def test_a_completed_label_still_wins_over_a_stale_prediction():
    """Re-prediction must never overwrite finished human work, whatever the
    version says."""
    images = [_image(1, "a")]
    labels = [SimpleNamespace(image_id=1, label_studio_project_id=73, completed=True)]
    needing = sut._select_images_needing_prediction(
        images, [_prediction(1, version=None)], labels
    )
    assert needing == []


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
        predictions=[_prediction(1)],
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


async def test_duplicate_frames_are_not_dispatched_as_work(monkeypatch):
    """Resolvers must mirror the cohort selectors, which gate on
    `is_canonical`.

    The same physical frames live under several dive rows — half of prod's
    image table is duplicate content — and `is_canonical` marks which copy is
    real. If a resolver dispatched a duplicate the selector had excluded, the
    per-image work would not match what the cohort promised, and (worse) the
    dive could never drain: no label row would ever appear for it, so the
    cohort predicate stays true forever. That is the prod dive 60 wedge.
    """
    fs = _make_fs(
        images=[
            _image(11, "c11", is_canonical=True),
            _image(12, "c12", is_canonical=False),
        ]
    )
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_laser_predict_inputs_activity, 1
    )

    dispatched = {img.checksum for img in result.images}
    assert "c11" in dispatched
    assert "c12" not in dispatched, "a duplicate frame was dispatched as work"

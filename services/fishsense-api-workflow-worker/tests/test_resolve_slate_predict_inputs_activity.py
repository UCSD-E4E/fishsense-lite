# pylint: disable=protected-access
"""Unit tests for resolve_slate_predict_inputs_activity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    resolve_slate_predict_inputs_activity as sut,
)

MARKER = sut.SLATE_CONTENT_MARKER


def _species(image_id, content):
    return SimpleNamespace(image_id=image_id, content_of_image=content)


def _pred(image_id):
    return SimpleNamespace(image_id=image_id)


def _label(image_id, *, completed, superseded):
    return SimpleNamespace(image_id=image_id, completed=completed, superseded=superseded)


def test_select_frames_needing_prediction():
    species = [
        _species(1, MARKER),   # fresh slate frame -> needs
        _species(2, MARKER),   # predicted -> skip
        _species(3, "Fish"),   # not a slate frame -> skip
        _species(4, MARKER),   # completed live label -> skip
        _species(5, MARKER),   # completed but superseded -> re-enters
    ]
    preds = [_pred(2)]
    labels = [
        _label(4, completed=True, superseded=False),
        _label(5, completed=True, superseded=True),
        _label(1, completed=False, superseded=False),  # placeholder: still needs
    ]
    assert set(sut._select_frames_needing_prediction(species, preds, labels)) == {1, 5}


def _make_fs():
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(
        return_value=SimpleNamespace(id=1, camera_id=7, dive_slate_id=99)
    )
    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(
        return_value=SimpleNamespace(
            camera_matrix=np.eye(3), distortion_coefficients=np.zeros(5)
        )
    )
    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(
        return_value=[
            SimpleNamespace(
                id=99, name="V-Slate 3", dpi=300, reference_points=[(0.0, 0.0), (1.0, 2.0)]
            )
        ]
    )
    fs.images = MagicMock()
    fs.images.get = AsyncMock(
        return_value=[
            SimpleNamespace(id=11, checksum="c11", is_canonical=True),
            SimpleNamespace(id=12, checksum="c12", is_canonical=True),
        ]
    )
    fs.labels = MagicMock()
    fs.labels.get_species_labels = AsyncMock(
        return_value=[_species(11, MARKER), _species(12, MARKER)]
    )
    fs.labels.get_slate_predictions = AsyncMock(return_value=[_pred(12)])
    fs.labels.get_dive_slate_labels = AsyncMock(return_value=[])
    return fs


@pytest.mark.asyncio
async def test_activity_builds_input_for_unpredicted_frames(monkeypatch):
    fs = _make_fs()
    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.resolve_slate_predict_inputs_activity, 1
    )

    assert result.dive_id == 1
    assert result.slate_id == 99
    assert result.slate_name == "V-Slate 3"
    assert result.dpi == 300
    assert result.template_points == [[0.0, 0.0], [1.0, 2.0]]
    # image 12 is already predicted -> only 11 is dispatched
    assert [i.image_id for i in result.images] == [11]
    assert result.images[0].checksum == "c11"

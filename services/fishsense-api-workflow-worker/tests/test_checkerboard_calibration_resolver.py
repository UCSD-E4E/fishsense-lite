"""Resolver for checkerboard laser calibration.

It answers "which frames are worth *trying*", and its image predicate has to
mirror the cohort selector's exactly. A resolver that selects differently from
its cohort is what stages a dive's raw `.ORF`s from the NAS every hour and
dispatches nothing — the failure CLAUDE.md calls out and that stage 5.1 has
already had.

Kept apart from `test_checkerboard_calibration_parent.py` on purpose: this
module imports numpy, and a test module that both imports numpy and defines a
`@workflow.defn` breaks Temporal's workflow sandbox, which re-imports the
module and cannot load a C extension twice in one process.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fishsense_shared import PerformCheckerboardCalibrationInput
from temporalio.testing import ActivityEnvironment

from fishsense_api_workflow_worker.activities import (
    resolve_checkerboard_calibration_inputs_activity as resolver_module,
)


def _laser_label(image_id, *, x=600.0, y=500.0, superseded=False):
    return SimpleNamespace(image_id=image_id, x=x, y=y, superseded=superseded)


def _image(image_id, *, is_canonical=True):
    return SimpleNamespace(
        id=image_id, checksum=f"{image_id:032d}", is_canonical=is_canonical
    )


def _make_fs(*, laser_labels, images, targets=None, dive=None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(
        return_value=(
            dive
            if dive is not None
            else SimpleNamespace(id=488, camera_id=9, calibration_target_id=4)
        )
    )

    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(
        return_value=SimpleNamespace(
            camera_matrix=np.array(
                [[1800.0, 0.0, 640.0], [0.0, 1800.0, 480.0], [0.0, 0.0, 1.0]]
            ),
            distortion_coefficients=np.zeros(5),
        )
    )

    fs.calibration_targets = MagicMock()
    fs.calibration_targets.get = AsyncMock(
        return_value=(
            targets
            if targets is not None
            else [
                SimpleNamespace(
                    id=4,
                    name="E4E Checkerboard",
                    rows=10,
                    cols=14,
                    square_size_m=0.0254,
                )
            ]
        )
    )

    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=images)
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=laser_labels)
    return fs


async def _resolve(fs, monkeypatch, dive_id=488):
    monkeypatch.setattr(resolver_module, "get_fs_client", lambda: fs)
    return await ActivityEnvironment().run(
        resolver_module.resolve_checkerboard_calibration_inputs_activity, dive_id
    )


@pytest.mark.asyncio
async def test_resolver_carries_the_board_geometry(monkeypatch):
    """The measured pitch reaches the child in the payload, read from the row.

    Resolved here rather than on the data-worker so a replayed child cannot
    pick up a different square size than the run was dispatched with — and
    the pitch is the only thing setting the scale of every length the
    resulting calibration will later produce.
    """
    resolved = await _resolve(
        _make_fs(
            laser_labels=[_laser_label(101), _laser_label(102)],
            images=[_image(101), _image(102)],
        ),
        monkeypatch,
    )

    assert isinstance(resolved, PerformCheckerboardCalibrationInput)
    assert (resolved.target_rows, resolved.target_cols) == (10, 14)
    assert resolved.square_size_m == 0.0254
    assert resolved.camera_id == 9
    assert [image.image_id for image in resolved.images] == [101, 102]
    assert [image.checksum for image in resolved.images] == [
        f"{101:032d}",
        f"{102:032d}",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label_kwargs", "why"),
    [
        ({"x": None, "y": None}, "populate-seeded placeholder, no dot"),
        ({"superseded": True}, "dead-lettered by the RANSAC validator"),
    ],
)
async def test_resolver_drops_unusable_laser_labels(monkeypatch, label_kwargs, why):
    """Mirrors the cohort's `has_live_laser_dot`, which is what
    `get_laser_label` filters on — nothing about `completed`."""
    resolved = await _resolve(
        _make_fs(
            laser_labels=[_laser_label(101), _laser_label(102, **label_kwargs)],
            images=[_image(101), _image(102)],
        ),
        monkeypatch,
    )

    assert [image.image_id for image in resolved.images] == [101], why


@pytest.mark.asyncio
async def test_resolver_drops_non_canonical_frames(monkeypatch):
    """Half of prod's image table is duplicate content and every cohort gates
    on `is_canonical`; a resolver that did not would dispatch work the cohort
    never promised — and two of the 2023.08.18 dives this stage serves are
    wholly or partly duplicate."""
    resolved = await _resolve(
        _make_fs(
            laser_labels=[_laser_label(101), _laser_label(102)],
            images=[_image(101), _image(102, is_canonical=False)],
        ),
        monkeypatch,
    )

    assert [image.image_id for image in resolved.images] == [101]


@pytest.mark.asyncio
async def test_resolver_refuses_a_dangling_target(monkeypatch):
    """A dive pointing at a target that no longer exists has no known scale."""
    fs = _make_fs(
        laser_labels=[_laser_label(101)],
        images=[_image(101)],
        dive=SimpleNamespace(id=488, camera_id=9, calibration_target_id=99),
    )

    with pytest.raises(ValueError, match="calibration_target_id=99 not found"):
        await _resolve(fs, monkeypatch)


@pytest.mark.asyncio
async def test_resolver_refuses_an_unlinked_dive(monkeypatch):
    fs = _make_fs(
        laser_labels=[_laser_label(101)],
        images=[_image(101)],
        dive=SimpleNamespace(id=488, camera_id=9, calibration_target_id=None),
    )

    with pytest.raises(ValueError, match="no calibration_target_id"):
        await _resolve(fs, monkeypatch)

"""A partial redraw must keep "image i of N" true for the whole cluster.

The stage 2 overlay writes the frame's position within its PREDICTION cluster
-- "image 2 of 7" -- and that position is a property of the cluster, not of
whichever subset happens to be rendered on a given run. The resolver emits only
the images that need work, so computing the position over that emitted list
answers a different question: redraw 3 of a 7-image cluster and they are
labelled 1/3, 2/3, 3/3 while their four siblings still read 4/7..7/7, at the
same object-store keys Label Studio presigns. The context the stage exists to
provide is destroyed, and nothing errors.

`needs_reprocess` makes this systematic -- a deliberate partial redraw is the
normal case -- but it is not new. Any image that becomes eligible after its
cluster was first processed hits it too: a laser validated after stage 1 ran,
or an orphan later assigned a cluster. Those resolve alone and render "1 of 1".

So the position travels with each image instead of being inferred from the
batch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from worker_tests_support.preprocess import (
    dive as _dive,
    image as _image,
    intrinsics as _intrinsics,
)

#: Three frames in one PREDICTION cluster.
_IDS = (1, 2, 3)
_SUMS = {i: chr(ord("a") + i) * 32 for i in _IDS}


def _laser(image_id: int):
    return LaserLabel(
        id=image_id, image_id=image_id, label_studio_task_id=10,
        label_studio_project_id=73, updated_at=None, completed=True,
        label_studio_json={}, user_id=None, superseded=False,
        needs_reprocess=False, x=5.0, y=6.0, label=None,
    )


def _species(image_id: int, *, needs_reprocess: bool):
    """A real (non-sentinel) species row, so the image is NOT eligible the
    ordinary way -- only the flag can reach it."""
    return SpeciesLabel(
        id=image_id, label_studio_task_id=10, label_studio_project_id=1,
        image_url=None, updated_at=None, completed=True, superseded=False,
        needs_reprocess=needs_reprocess, grouping=None,
        top_three_photos_of_group=None, slate_upside_down=None, laser_x=None,
        laser_y=None, laser_label=None, content_of_image=None,
        fish_measurable_category=None, fish_angle_category=None,
        fish_curved_category=None, label_studio_json={}, image_id=image_id,
        user_id=None,
    )


def _cluster(image_ids):
    c = MagicMock()
    c.image_ids = list(image_ids)
    return c


def _fs(*, flagged_ids):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=_dive())
    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=_intrinsics())
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=[_image(i, _SUMS[i]) for i in _IDS])
    fs.images.get_clusters = AsyncMock(return_value=[_cluster(_IDS)])
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=[_laser(i) for i in _IDS])
    fs.labels.get_species_labels = AsyncMock(
        return_value=[_species(i, needs_reprocess=i in flagged_ids) for i in _IDS]
    )
    return fs


async def _resolve(monkeypatch, flagged_ids):
    from fishsense_api_workflow_worker.activities import (
        resolve_species_preprocess_inputs_activity as sut,
    )

    monkeypatch.setattr(sut, "get_fs_client", lambda: _fs(flagged_ids=flagged_ids))
    return await ActivityEnvironment().run(
        sut.resolve_species_preprocess_inputs_activity, 42
    )


class TestClusterPositions:
    async def test_middle_image_alone_keeps_its_true_position(self, monkeypatch):
        result = await _resolve(monkeypatch, {2})

        assert result.cluster_members is not None
        members = [m for c in result.cluster_members for m in c]
        assert [m.checksum for m in members] == [_SUMS[2]]
        assert members[0].cluster_index == 2, "second of three, not first of one"
        assert members[0].cluster_size == 3

    @pytest.mark.parametrize("flagged", [{1}, {2}, {3}, {1, 3}, {1, 2, 3}])
    async def test_every_subset_agrees_with_the_full_cluster(
        self, monkeypatch, flagged
    ):
        """Whatever is redrawn, each frame reports the same i/N it would have
        had in a full pass. That is the invariant the siblings' JPEGs encode."""
        result = await _resolve(monkeypatch, flagged)

        members = [m for c in result.cluster_members for m in c]
        assert {m.checksum for m in members} == {_SUMS[i] for i in flagged}
        for member in members:
            image_id = next(i for i in _IDS if _SUMS[i] == member.checksum)
            assert member.cluster_index == image_id, "1-based position in the cluster"
            assert member.cluster_size == 3

    async def test_clusters_still_carries_only_the_work(self, monkeypatch):
        """The old field keeps its meaning so an older data-worker reading it
        during a rolling deploy redraws the same set, just with the i/N it
        always had."""
        result = await _resolve(monkeypatch, {2})

        assert result.clusters == [[_SUMS[2]]]

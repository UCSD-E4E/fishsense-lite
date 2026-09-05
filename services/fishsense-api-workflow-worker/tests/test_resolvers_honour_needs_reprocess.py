# pylint: disable=unused-argument,protected-access
"""Species / head/tail / slate resolvers must return flagged images.

The selector and the resolver are two halves of one predicate. Stage 0.1 has
honoured `needs_reprocess` on both sides for a while; the other three honoured
it on neither. Now that their cohorts select flagged dives, their resolvers
have to return the flagged images too -- otherwise the parent picks the dive,
wakes the data-worker, stages the dive's raw `.ORF`s from the NAS, resolves
zero images, cleans up and does the whole thing again next hour, forever,
while every higher-id dive waits behind it.

CLAUDE.md states the rule ("Resolver activities mirror the same predicate")
and prod has paid for breaking it more than once.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_sdk.models.laser_label import LaserLabel
from worker_tests_support.preprocess import (
    dive as _dive,
    image as _image,
    intrinsics as _intrinsics,
)

_CHECKSUM = "a" * 32


def _valid_laser(image_id=1):
    """A laser label that passes the validity gate species and head/tail
    cascade from: completed, not superseded, both coordinates populated. The
    shared helper hardcodes x/y to None, which would fail that gate."""
    return LaserLabel(
        id=1, image_id=image_id, label_studio_task_id=10,
        label_studio_project_id=73, updated_at=None, completed=True,
        label_studio_json={}, user_id=None, superseded=False,
        needs_reprocess=False, x=5.0, y=6.0, label=None,
    )


def _headtail(*, needs_reprocess):
    return HeadTailLabel(
        id=1, label_studio_task_id=10, label_studio_project_id=1,
        head_x=1.0, head_y=2.0, tail_x=3.0, tail_y=4.0, updated_at=None,
        superseded=False, completed=True, needs_reprocess=needs_reprocess,
        label_studio_json={}, image_id=1, user_id=None,
    )


def _species(*, needs_reprocess, content_of_image=None):
    return SpeciesLabel(
        id=1, label_studio_task_id=10, label_studio_project_id=1,
        image_url=None, updated_at=None, completed=True, superseded=False,
        needs_reprocess=needs_reprocess, grouping=None,
        top_three_photos_of_group=None, slate_upside_down=None,
        laser_x=None, laser_y=None, laser_label=None,
        content_of_image=content_of_image, fish_measurable_category=None,
        fish_angle_category=None, fish_curved_category=None,
        label_studio_json={}, image_id=1, user_id=None,
    )


def _slate(*, needs_reprocess):
    return DiveSlateLabel(
        id=1, label_studio_task_id=10, label_studio_project_id=1,
        image_url=None, upside_down=None, reference_points=None,
        slate_rectangle=None, skipped_points=None, updated_at=None,
        completed=True, superseded=False, needs_reprocess=needs_reprocess,
        label_studio_json={}, image_id=1, user_id=None,
    )


def _fs(*, labels, kind, clusters=None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.dives = MagicMock()
    dive = _dive()
    dive.dive_slate_id = 1          # stage 9 refuses a dive without one
    fs.dives.get = AsyncMock(return_value=dive)
    # Stage 9 also needs the slate template it will composite into the overlay.
    slate = MagicMock()
    slate.id = 1
    slate.dpi = 300
    slate.reference_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(return_value=[slate])
    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=_intrinsics())
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=[_image(1, _CHECKSUM)])
    fs.images.get_clusters = AsyncMock(return_value=clusters or [])
    fs.labels = MagicMock()
    # Every image carries a *valid* laser label: the gate species and head/tail
    # cascade from. Without it the flag could never be reached.
    fs.labels.get_laser_labels = AsyncMock(return_value=[_valid_laser()])
    fs.labels.get_headtail_labels = AsyncMock(
        return_value=labels if kind == "headtail" else []
    )
    fs.labels.get_species_labels = AsyncMock(
        return_value=labels if kind == "species" else _slate_marker()
    )
    fs.labels.get_dive_slate_labels = AsyncMock(
        return_value=labels if kind == "dive-slate" else []
    )
    return fs


def _slate_marker():
    """Stage 9 finds its frames by the species taxonomy marker."""
    return [_species(needs_reprocess=False, content_of_image="Slate, Laser on slate")]


def _cluster():
    c = MagicMock()
    c.image_ids = [1]
    return c


class TestHeadtail:
    async def test_flagged_image_is_returned_though_it_is_already_labelled(
        self, monkeypatch
    ):
        from fishsense_api_workflow_worker.activities import (
            resolve_headtail_preprocess_inputs_activity as sut,
        )

        labels = [_headtail(needs_reprocess=True)]
        monkeypatch.setattr(sut, "get_fs_client", lambda: _fs(labels=labels, kind="headtail"))
        result = await ActivityEnvironment().run(
            sut.resolve_headtail_preprocess_inputs_activity, 42
        )
        assert result.image_checksums == [_CHECKSUM]

    async def test_unflagged_labelled_image_is_still_excluded(self, monkeypatch):
        from fishsense_api_workflow_worker.activities import (
            resolve_headtail_preprocess_inputs_activity as sut,
        )

        labels = [_headtail(needs_reprocess=False)]
        monkeypatch.setattr(sut, "get_fs_client", lambda: _fs(labels=labels, kind="headtail"))
        result = await ActivityEnvironment().run(
            sut.resolve_headtail_preprocess_inputs_activity, 42
        )
        assert result.image_checksums == []


class TestSpecies:
    async def test_flagged_image_is_returned_though_it_is_already_labelled(
        self, monkeypatch
    ):
        from fishsense_api_workflow_worker.activities import (
            resolve_species_preprocess_inputs_activity as sut,
        )

        labels = [_species(needs_reprocess=True)]
        monkeypatch.setattr(
            sut,
            "get_fs_client",
            lambda: _fs(labels=labels, kind="species", clusters=[_cluster()]),
        )
        result = await ActivityEnvironment().run(
            sut.resolve_species_preprocess_inputs_activity, 42
        )
        assert [c for c in result.clusters] == [[_CHECKSUM]]

    async def test_unflagged_labelled_image_is_still_excluded(self, monkeypatch):
        from fishsense_api_workflow_worker.activities import (
            resolve_species_preprocess_inputs_activity as sut,
        )

        labels = [_species(needs_reprocess=False)]
        monkeypatch.setattr(
            sut,
            "get_fs_client",
            lambda: _fs(labels=labels, kind="species", clusters=[_cluster()]),
        )
        result = await ActivityEnvironment().run(
            sut.resolve_species_preprocess_inputs_activity, 42
        )
        assert result.clusters == []


class TestSlate:
    async def test_flagged_image_is_returned_though_it_is_already_labelled(
        self, monkeypatch
    ):
        from fishsense_api_workflow_worker.activities import (
            resolve_slate_preprocess_inputs_activity as sut,
        )

        labels = [_slate(needs_reprocess=True)]
        monkeypatch.setattr(sut, "get_fs_client", lambda: _fs(labels=labels, kind="dive-slate"))
        result = await ActivityEnvironment().run(
            sut.resolve_slate_preprocess_inputs_activity, 42
        )
        assert result.image_checksums == [_CHECKSUM]

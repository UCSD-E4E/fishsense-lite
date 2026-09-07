# pylint: disable=unused-argument,protected-access
"""A raised flag must always end up lowered, and must always reach its image.

Every gap between "the selector picks this dive" and "the resolver finds work
for it" is a wedge: the parent stages the dive's raw `.ORF`s from the NAS,
resolves nothing, returns, and the selector picks the same dive again next
hour, forever, starving every higher-id dive behind it.

Three ways that gap opened, all found in review:

  * a flagged image in no PREDICTION cluster (species orphans) was filtered out
    by the cluster walk *and* by the orphan branch;
  * a flagged image whose laser was superseded after flagging fell out of the
    resolver, because the resolver's flag branch was gated on a currently-valid
    laser while the cohort's was not;
  * the early "no work resolved" return sits before the clear step, so any flag
    that resolves to nothing is never lowered.

The last one is the backstop: even if a flag reaches no image, lowering it
keeps the cohort drainable. Dropping a flag silently is worse than wedging only
in the sense that it is quiet -- so it is logged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from fishsense_api_sdk.models.headtail_label import HeadTailLabel
from fishsense_api_sdk.models.laser_label import LaserLabel
from fishsense_api_sdk.models.species_label import SpeciesLabel
from worker_tests_support.preprocess import (
    dive as _dive,
    image as _image,
    intrinsics as _intrinsics,
)

_CHECKSUM = "b" * 32


def _laser(*, valid: bool):
    return LaserLabel(
        id=1, image_id=1, label_studio_task_id=10, label_studio_project_id=73,
        updated_at=None, completed=valid, label_studio_json={}, user_id=None,
        superseded=not valid, needs_reprocess=False,
        x=5.0 if valid else None, y=6.0 if valid else None, label=None,
    )


def _species(*, needs_reprocess, content_of_image=None):
    return SpeciesLabel(
        id=1, label_studio_task_id=10, label_studio_project_id=1, image_url=None,
        updated_at=None, completed=True, superseded=False,
        needs_reprocess=needs_reprocess, grouping=None,
        top_three_photos_of_group=None, slate_upside_down=None, laser_x=None,
        laser_y=None, laser_label=None, content_of_image=content_of_image,
        fish_measurable_category=None, fish_angle_category=None,
        fish_curved_category=None, label_studio_json={}, image_id=1, user_id=None,
    )


def _headtail(*, needs_reprocess):
    return HeadTailLabel(
        id=1, label_studio_task_id=10, label_studio_project_id=1, head_x=1.0,
        head_y=2.0, tail_x=3.0, tail_y=4.0, updated_at=None, superseded=False,
        completed=True, needs_reprocess=needs_reprocess, label_studio_json={},
        image_id=1, user_id=None,
    )


def _slate(*, needs_reprocess):
    return DiveSlateLabel(
        id=1, label_studio_task_id=10, label_studio_project_id=1, image_url=None,
        upside_down=None, reference_points=None, slate_rectangle=None,
        skipped_points=None, updated_at=None, completed=True, superseded=False,
        needs_reprocess=needs_reprocess, label_studio_json={}, image_id=1,
        user_id=None,
    )


def _fs(*, laser_valid, species=None, headtail=None, slate=None, clusters=None):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    dive = _dive()
    dive.dive_slate_id = 1
    fs.dives = MagicMock()
    fs.dives.get = AsyncMock(return_value=dive)
    fs.cameras = MagicMock()
    fs.cameras.get_intrinsics = AsyncMock(return_value=_intrinsics())
    fs.images = MagicMock()
    fs.images.get = AsyncMock(return_value=[_image(1, _CHECKSUM)])
    fs.images.get_clusters = AsyncMock(return_value=clusters or [])
    tpl = MagicMock()
    tpl.id, tpl.dpi = 1, 300
    tpl.reference_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    fs.dive_slates = MagicMock()
    fs.dive_slates.get = AsyncMock(return_value=[tpl])
    fs.labels = MagicMock()
    fs.labels.get_laser_labels = AsyncMock(return_value=[_laser(valid=laser_valid)])
    fs.labels.get_species_labels = AsyncMock(return_value=species or [])
    fs.labels.get_headtail_labels = AsyncMock(return_value=headtail or [])
    fs.labels.get_dive_slate_labels = AsyncMock(return_value=slate or [])
    return fs


class TestSpeciesOrphans:
    async def test_flagged_orphan_is_resolved(self, monkeypatch):
        """No PREDICTION cluster contains it, so only the orphan branch can
        reach it -- and that branch ignored the flag entirely."""
        from fishsense_api_workflow_worker.activities import (
            resolve_species_preprocess_inputs_activity as sut,
        )

        monkeypatch.setattr(
            sut,
            "get_fs_client",
            lambda: _fs(
                laser_valid=True,
                species=[_species(needs_reprocess=True)],
                clusters=[],
            ),
        )
        result = await ActivityEnvironment().run(
            sut.resolve_species_preprocess_inputs_activity, 42
        )
        assert result.clusters == [[_CHECKSUM]]


class TestSupersededLaserStillResolves:
    async def test_headtail_flagged_image_resolves_though_its_laser_was_superseded(
        self, monkeypatch
    ):
        """The cohort's flag branch has no laser gate, so the resolver's must
        not either -- or a laser superseded after flagging wedges the dive."""
        from fishsense_api_workflow_worker.activities import (
            resolve_headtail_preprocess_inputs_activity as sut,
        )

        monkeypatch.setattr(
            sut,
            "get_fs_client",
            lambda: _fs(laser_valid=False, headtail=[_headtail(needs_reprocess=True)]),
        )
        result = await ActivityEnvironment().run(
            sut.resolve_headtail_preprocess_inputs_activity, 42
        )
        assert result.image_checksums == [_CHECKSUM]

    async def test_slate_flagged_image_resolves_without_the_content_marker(
        self, monkeypatch
    ):
        """Stage 9 normally finds frames by the species taxonomy marker. A
        flagged frame must not need it: the cohort's flag branch does not."""
        from fishsense_api_workflow_worker.activities import (
            resolve_slate_preprocess_inputs_activity as sut,
        )

        monkeypatch.setattr(
            sut,
            "get_fs_client",
            lambda: _fs(
                laser_valid=True,
                species=[_species(needs_reprocess=False, content_of_image="Fish, Hogfish")],
                slate=[_slate(needs_reprocess=True)],
            ),
        )
        result = await ActivityEnvironment().run(
            sut.resolve_slate_preprocess_inputs_activity, 42
        )
        assert result.image_checksums == [_CHECKSUM]

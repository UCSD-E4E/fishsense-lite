# pylint: disable=unused-argument
"""Unit tests for update_dive_image_groups_activity (stage 6.1 port).

Three things this file pins down:
  1. `regroup_by_species_labels` honors the "Part of previous group" /
     "Not part of current group" boundary contract from the notebook.
  2. The activity refuses to re-create LABEL_STUDIO clusters when any
     already exist (idempotent skip).
  3. The activity POSTs one cluster per emitted group with
     `data_source=LABEL_STUDIO`.
"""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.species_label import SpeciesLabel
from fishsense_api_workflow_worker.activities import (
    update_dive_image_groups_activity as sut,
)


def _label(image_id: int, *, grouping: str | None = None) -> SpeciesLabel:
    return SpeciesLabel(
        id=None,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=70,
        image_url=None,
        updated_at=None,
        completed=True,
        grouping=grouping,
        top_three_photos_of_group=None,
        slate_upside_down=None,
        laser_x=None,
        laser_y=None,
        laser_label=None,
        content_of_image=None,
        fish_measurable_category=None,
        fish_angle_category=None,
        fish_curved_category=None,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def _prediction_cluster(cluster_id: int, image_ids: List[int]) -> DiveFrameCluster:
    return DiveFrameCluster(
        id=cluster_id,
        image_ids=image_ids,
        data_source=DataSource.PREDICTION,
        updated_at=None,
        dive_id=1,
        fish_id=None,
    )


# ----------------------------- pure regrouping ----------------------------


def test_regroup_first_label_of_each_cluster_starts_new_group_by_default():
    clusters = [
        _prediction_cluster(1, [101, 101, 102]),
        _prediction_cluster(2, [103, 104]),
    ]
    labels = {
        101: _label(101),
        102: _label(102),
        103: _label(103),
        104: _label(104),
    }

    groups = sut.regroup_by_species_labels(clusters, labels)

    # First cluster's first label opens group A; second cluster's first
    # label opens group B (no continuity marker).
    assert groups == [[101, 101, 102], [103, 104]]


def test_regroup_part_of_previous_group_continues_across_cluster_boundary():
    clusters = [
        _prediction_cluster(1, [101, 102]),
        _prediction_cluster(2, [103, 104]),
    ]
    labels = {
        101: _label(101),
        102: _label(102),
        103: _label(103, grouping="Part of previous group"),
        104: _label(104),
    }

    groups = sut.regroup_by_species_labels(clusters, labels)

    # Cluster 2's first label continues into the previous group; 104
    # then starts a new group (idx != 0, default grouping).
    assert groups == [[101, 102, 103, 104]]


def test_regroup_not_part_of_current_group_breaks_mid_cluster():
    clusters = [_prediction_cluster(1, [101, 102, 103, 104])]
    labels = {
        101: _label(101),
        102: _label(102),
        103: _label(103, grouping="Not part of current group"),
        104: _label(104),
    }

    groups = sut.regroup_by_species_labels(clusters, labels)

    # 103 flushes [101, 102] and starts a new group containing itself
    # and 104.
    assert groups == [[101, 102], [103, 104]]


def test_regroup_skips_image_ids_without_a_species_label():
    clusters = [_prediction_cluster(1, [101, 102, 103])]
    labels = {101: _label(101), 103: _label(103)}

    groups = sut.regroup_by_species_labels(clusters, labels)

    # 102 has no label entry — quietly skipped, doesn't open a group.
    assert groups == [[101, 103]]


def test_regroup_empty_inputs():
    assert sut.regroup_by_species_labels([], {}) == []
    assert sut.regroup_by_species_labels(
        [_prediction_cluster(1, [101])], {}
    ) == []


def test_regroup_first_cluster_starts_with_part_of_previous_does_not_open_extra_group():
    # Edge: dive starts with "Part of previous group" — there is no
    # previous group, but the marker means "don't insert a boundary."
    # The label still gets appended to the (initially empty) current
    # group, producing exactly one group.
    clusters = [_prediction_cluster(1, [101, 102])]
    labels = {
        101: _label(101, grouping="Part of previous group"),
        102: _label(102),
    }

    groups = sut.regroup_by_species_labels(clusters, labels)

    assert groups == [[101, 102]]


# ------------------------------- activity --------------------------------


def _make_fs_client(
    *,
    label_studio_clusters: List[DiveFrameCluster],
    prediction_clusters: List[DiveFrameCluster],
    species_labels: List[SpeciesLabel],
):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)

    async def _get_clusters(dive_id, data_source):
        if data_source == DataSource.LABEL_STUDIO.value:
            return label_studio_clusters
        if data_source == DataSource.PREDICTION.value:
            return prediction_clusters
        return []

    fs.images = MagicMock()
    fs.images.get_clusters = AsyncMock(side_effect=_get_clusters)
    fs.images.post_cluster = AsyncMock(return_value=999)

    fs.labels = MagicMock()
    fs.labels.get_species_labels = AsyncMock(return_value=species_labels)
    return fs


@pytest.mark.asyncio
async def test_activity_skips_when_label_studio_clusters_already_exist(monkeypatch):
    existing = [
        DiveFrameCluster(
            id=42,
            image_ids=[101, 102],
            data_source=DataSource.LABEL_STUDIO,
            updated_at=None,
            dive_id=1,
            fish_id=None,
        )
    ]
    fs = _make_fs_client(
        label_studio_clusters=existing,
        prediction_clusters=[_prediction_cluster(1, [101, 102])],
        species_labels=[_label(101), _label(102)],
    )

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.update_dive_image_groups_activity, 1
    )

    assert result.skipped_already_grouped is True
    assert result.new_clusters_created == 0
    fs.images.post_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_activity_creates_one_cluster_per_group(monkeypatch):
    fs = _make_fs_client(
        label_studio_clusters=[],
        prediction_clusters=[
            _prediction_cluster(1, [101, 102]),
            _prediction_cluster(2, [103, 104]),
        ],
        species_labels=[
            _label(101),
            _label(102),
            _label(103, grouping="Part of previous group"),
            _label(104, grouping="Not part of current group"),
        ],
    )

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.update_dive_image_groups_activity, 7
    )

    # Expected groups:
    # - [101, 102, 103]  (103 continues previous group)
    # - [104]            (104 explicitly breaks into a new group)
    assert result.skipped_already_grouped is False
    assert result.new_clusters_created == 2
    assert result.species_labels_seen == 4
    assert fs.images.post_cluster.await_count == 2

    posted_clusters = [c.args[1] for c in fs.images.post_cluster.await_args_list]
    posted_image_ids = [cluster.image_ids for cluster in posted_clusters]
    assert posted_image_ids == [[101, 102, 103], [104]]
    assert all(
        cluster.data_source == DataSource.LABEL_STUDIO
        for cluster in posted_clusters
    )
    assert all(cluster.dive_id == 7 for cluster in posted_clusters)


@pytest.mark.asyncio
async def test_activity_no_groups_is_not_a_skip(monkeypatch):
    # No prediction clusters → nothing to group, but this is "no work
    # possible" not "already done." Tests should be able to tell them
    # apart via skipped_already_grouped.
    fs = _make_fs_client(
        label_studio_clusters=[],
        prediction_clusters=[],
        species_labels=[],
    )

    monkeypatch.setattr(sut, "get_fs_client", lambda: fs)

    result = await ActivityEnvironment().run(
        sut.update_dive_image_groups_activity, 9
    )

    assert result.skipped_already_grouped is False
    assert result.new_clusters_created == 0
    fs.images.post_cluster.assert_not_called()

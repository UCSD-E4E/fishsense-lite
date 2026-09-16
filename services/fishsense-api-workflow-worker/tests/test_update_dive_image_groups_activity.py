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
        superseded=False,
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
    assert not sut.regroup_by_species_labels([], {})
    assert not sut.regroup_by_species_labels(
        [_prediction_cluster(1, [101])], {}
    )


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


# --- sentinel species rows must not drive grouping --------------------------
#
# A `SpeciesLabel` with `label_studio_project_id IS NULL` is a SENTINEL: not a
# labeler's answer. Every cohort selector in `dive_cohort_controller` reads
# "no non-sentinel row" as "unlabelled", which is what lets a bulk import of
# stored judgements sit in the database without taking dives out of the
# labelling flow.
#
# This activity did not honour that. It built its lookup from every species row
# the dive has, with no project and no superseded filter, so sentinels were read
# as human answers. Measured in prod 2026-09-16 on dive 5
# (`Hogfish01_MolHITW_0926_080323`): image 1399 had no real species label --
# populate skipped it because its stage-2 JPEG was missing -- so its only row
# was an imported sentinel with `grouping = NULL`. 1399 LEADS a PREDICTION
# cluster, so a new group started there and ONE hogfish became TWO Fish rows
# with 4 and 7 measurements. `species_labels_seen: 20` on a 16-task project was
# the tell.
#
# Second failure mode: the lookup was a dict comprehension keyed on image_id,
# so where a frame held both a real row and a sentinel, whichever came last
# won -- a sentinel could silently override a human answer. Dive 5 had three
# such frames.


def _sentinel(image_id: int, *, grouping: str | None = None) -> SpeciesLabel:
    """An imported judgement: carries a species, belongs to no LS project."""
    label = _label(image_id, grouping=grouping)
    return label.model_copy(
        update={
            "label_studio_project_id": None,
            "label_studio_task_id": None,
            "completed": False,
            "content_of_image": "Fish, Hogfish (Lachnolaimus maximus)",
        }
    )


def test_a_frame_whose_only_species_row_is_a_sentinel_is_not_grouped():
    """No human judged that frame, so it must not join a measurement cluster
    and must not start one either."""
    chosen = sut.select_species_label_per_image([_sentinel(5)])
    assert not chosen


def test_a_sentinel_never_overrides_a_real_answer_whatever_the_order():
    real = _label(5, grouping="Part of previous group")
    sentinel = _sentinel(5)
    for order in ([real, sentinel], [sentinel, real]):
        chosen = sut.select_species_label_per_image(order)
        assert chosen[5].label_studio_project_id == 70
        assert chosen[5].grouping == "Part of previous group"


def test_a_superseded_row_is_ignored():
    """`superseded` is this repo's dead-letter for every label kind; a
    dead-lettered answer must not decide a grouping."""
    dead = _label(5, grouping="Part of previous group").model_copy(
        update={"superseded": True}
    )
    assert not sut.select_species_label_per_image([dead])


def test_a_live_row_wins_over_a_superseded_one():
    dead = _label(5, grouping="Not part of current group").model_copy(
        update={"superseded": True}
    )
    live = _label(5, grouping="Part of previous group")
    for order in ([dead, live], [live, dead]):
        chosen = sut.select_species_label_per_image(order)
        assert chosen[5].grouping == "Part of previous group"


def test_the_choice_among_several_real_rows_is_deterministic():
    """A frame can carry rows in two real projects (the per-dive project plus a
    grandfathered one). Last-wins over an unordered list made the grouping
    depend on API row order; the highest id -- most recently written -- is a
    stated rule instead."""
    older = _label(5, grouping="Not part of current group").model_copy(
        update={"id": 100, "label_studio_project_id": 70}
    )
    newer = _label(5, grouping="Part of previous group").model_copy(
        update={"id": 200, "label_studio_project_id": 99}
    )
    for order in ([older, newer], [newer, older]):
        chosen = sut.select_species_label_per_image(order)
        assert chosen[5].id == 200


def test_the_prod_dive_5_shape_yields_one_group_not_two():
    """The regression this exists for, end to end through the real regrouper.

    Eight PREDICTION clusters; every cluster-leading frame carries
    "Part of previous group" EXCEPT 1399, whose only row is a sentinel with
    NULL grouping. Before the fix that split the run in two.
    """
    clusters = [
        _prediction_cluster(1, [1393, 1394]),
        _prediction_cluster(2, [1395, 1396]),
        _prediction_cluster(3, [1397, 1398]),
        _prediction_cluster(4, [1399, 1400]),
        _prediction_cluster(5, [1401, 1402]),
        _prediction_cluster(6, [1403, 1404]),
        _prediction_cluster(7, [1405, 1406, 1407]),
        _prediction_cluster(8, [1408, 1409]),
    ]
    cont = "Part of previous group"
    labels = [
        _label(1393), _label(1394),
        _label(1395, grouping=cont), _label(1396),
        _label(1397, grouping=cont), _label(1398),
        _sentinel(1399),                      # <- no human answer for this frame
        _label(1400, grouping=cont),
        _label(1401, grouping=cont), _label(1402),
        _label(1403, grouping=cont), _label(1404),
        _label(1405, grouping=cont), _label(1406), _label(1407),
        _label(1408, grouping=cont), _label(1409),
    ]

    groups = sut.regroup_by_species_labels(
        clusters, sut.select_species_label_per_image(labels)
    )

    assert len(groups) == 1, f"one hogfish must be one group, got {groups}"
    assert 1399 not in groups[0], "an unjudged frame must not be measured"
    assert len(groups[0]) == 16

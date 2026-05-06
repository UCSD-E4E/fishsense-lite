# pylint: disable=unused-argument
"""Unit tests for persist_dive_frame_clusters_activity."""

from __future__ import annotations

from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_workflow_worker.activities import (
    persist_dive_frame_clusters_activity as sut,
)


def _make_fs(post_calls: List):
    fs = MagicMock()
    fs.__aenter__ = AsyncMock(return_value=fs)
    fs.__aexit__ = AsyncMock(return_value=None)
    fs.images = MagicMock()

    async def post_cluster(dive_id, cluster):
        post_calls.append((dive_id, list(cluster.image_ids), cluster.data_source))
        return 1

    fs.images.post_cluster = AsyncMock(side_effect=post_cluster)
    return fs


@pytest.mark.asyncio
async def test_posts_one_prediction_cluster_per_id_list(monkeypatch):
    post_calls: List = []
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(post_calls))

    posted = await ActivityEnvironment().run(
        sut.persist_dive_frame_clusters_activity,
        42,
        [[1, 2, 3], [11, 12]],
    )

    assert posted == 2
    assert post_calls == [
        (42, [1, 2, 3], DataSource.PREDICTION),
        (42, [11, 12], DataSource.PREDICTION),
    ]


@pytest.mark.asyncio
async def test_skips_empty_inner_clusters(monkeypatch):
    """Defensive: HDBSCAN output should never include an empty inner
    list, but a future change to the activity could; the persist
    activity must not POST a zero-image cluster (the API would 400 or
    silently insert garbage)."""
    post_calls: List = []
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(post_calls))

    posted = await ActivityEnvironment().run(
        sut.persist_dive_frame_clusters_activity,
        42,
        [[1, 2, 3], [], [11]],
    )

    assert posted == 2
    image_id_lists = [call[1] for call in post_calls]
    assert [1, 2, 3] in image_id_lists
    assert [11] in image_id_lists
    assert [] not in image_id_lists


@pytest.mark.asyncio
async def test_returns_zero_when_no_clusters(monkeypatch):
    post_calls: List = []
    monkeypatch.setattr(sut, "get_fs_client", lambda: _make_fs(post_calls))

    posted = await ActivityEnvironment().run(
        sut.persist_dive_frame_clusters_activity, 42, []
    )

    assert posted == 0
    assert post_calls == []

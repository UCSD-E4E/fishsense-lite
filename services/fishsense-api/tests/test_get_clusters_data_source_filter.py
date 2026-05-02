"""Regression: GET /dives/{id}/images/clusters/{ds} must filter clusters by
data_source.

The bug: the cluster query was scoped only by dive_id, while the mapping
query was scoped by (dive_id, data_source). On a dive carrying both
PREDICTION and LABEL_STUDIO clusters, every LABEL_STUDIO cluster ended up
in the cluster list with no entry in the PREDICTION-only mapping dict,
which raised KeyError on lookup. The api-worker's stage-2 selector
activity retried the resulting 500 into its 10-min schedule_to_close
wall in prod (cluster id 2732 on dive 279).
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    # FK-less in-memory sqlite is enough — we're testing the controller's
    # query composition, not referential integrity. Bypass the prod
    # Database class because it hardcodes pool_size/max_overflow/pool_timeout
    # which the sqlite/StaticPool combo rejects.
    # Importing fishsense_api.database wires every model into SQLModel.metadata
    # so create_all sees them.
    from sqlalchemy.ext.asyncio import create_async_engine  # pylint: disable=import-outside-toplevel

    import fishsense_api.database  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def test_get_clusters_returns_only_requested_data_source(session):
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        get_clusters,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )

    dive_id = 279

    pred_a = DiveFrameCluster(dive_id=dive_id, data_source=DataSource.PREDICTION)
    pred_b = DiveFrameCluster(dive_id=dive_id, data_source=DataSource.PREDICTION)
    # The exact bug shape: a LABEL_STUDIO cluster on the same dive that
    # has zero PREDICTION mappings (because LABEL_STUDIO mappings are
    # scoped to the LS data source, which the controller's mapping query
    # already filters out).
    ls_orphan = DiveFrameCluster(dive_id=dive_id, data_source=DataSource.LABEL_STUDIO)
    # Sanity sibling: a different dive must not leak in either.
    other_dive = DiveFrameCluster(dive_id=dive_id + 1, data_source=DataSource.PREDICTION)

    session.add_all([pred_a, pred_b, ls_orphan, other_dive])
    await session.flush()

    session.add_all(
        [
            DiveFrameClusterImageMapping(dive_frame_cluster_id=pred_a.id, image_id=11),
            DiveFrameClusterImageMapping(dive_frame_cluster_id=pred_a.id, image_id=12),
            DiveFrameClusterImageMapping(dive_frame_cluster_id=pred_b.id, image_id=21),
            DiveFrameClusterImageMapping(
                dive_frame_cluster_id=ls_orphan.id, image_id=31
            ),
            DiveFrameClusterImageMapping(
                dive_frame_cluster_id=other_dive.id, image_id=99
            ),
        ]
    )
    await session.flush()

    result = await get_clusters(
        dive_id=dive_id, data_source=DataSource.PREDICTION, session=session
    )

    assert result is not None
    returned_ids = {c.id for c in result}
    assert returned_ids == {pred_a.id, pred_b.id}

    by_id = {c.id: c for c in result}
    assert set(by_id[pred_a.id].image_ids) == {11, 12}
    assert by_id[pred_b.id].image_ids == [21]
    for c in result:
        assert c.data_source == DataSource.PREDICTION


async def test_get_clusters_handles_cluster_with_no_mappings(session):
    """Defensive: a cluster of the requested data_source with zero rows in
    the mapping table must surface as image_ids=[], not KeyError."""
    from fishsense_api.controllers.image_controller import (  # pylint: disable=import-outside-toplevel
        get_clusters,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )

    empty = DiveFrameCluster(dive_id=42, data_source=DataSource.PREDICTION)
    session.add(empty)
    await session.flush()

    result = await get_clusters(
        dive_id=42, data_source=DataSource.PREDICTION, session=session
    )

    assert result is not None
    assert len(result) == 1
    assert result[0].id == empty.id
    assert result[0].image_ids == []

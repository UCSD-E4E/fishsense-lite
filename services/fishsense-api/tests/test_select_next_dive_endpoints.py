# pylint: disable=too-many-lines
"""Tests for the per-stage `GET /api/v1/dives/select-next/...` endpoints.

These collapse the api-workflow-worker's previously O(N) selector
activities (one HTTP call per HIGH-priority dive) into a single
SELECT … LIMIT 1. The cohort definitions live in CLAUDE.md; this file
pins them down at the SQL layer.

Uses the same FK-less in-memory sqlite fixture as
test_get_clusters_data_source_filter.py — we're testing query
composition, not referential integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


def _dive(dive_id: int, *, priority="HIGH", dive_slate_id=None, calibration_dive_id=None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority[priority],
        dive_slate_id=dive_slate_id,
        calibration_dive_id=calibration_dive_id,
    )


def _image(image_id: int, dive_id: int):
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"img-{image_id:032d}"[:32],
        dive_id=dive_id,
    )


# ---------- stage 0.1: laser-preprocessing ----------
#
# Cohort is "HIGH + has at least one image without ANY LaserLabel row
# (in any project)" — once populate seeds even an incomplete row, the
# image's preprocessed JPEG is on the file-exchange and the dive
# drops out. The earlier "no completed label" predicate kept dives in
# the cohort indefinitely after populate seeded incomplete sentinel
# rows, re-staging raw `.ORF`s from NAS every hour for no benefit.
# See dive_controller.py docstring for the rationale. A dive with
# zero images is excluded by the same `EXISTS (image without ...)`
# predicate, which matches the behavior at the resolver level
# (no images → no work).


async def test_laser_preprocessing_picks_lowest_high_priority_with_unlabeled_images(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    # dives 1 and 2 both have an unlabeled image; lowest id wins.
    # dive 3 is LOW priority -> excluded.
    session.add_all(
        [
            _dive(2, priority="HIGH"),
            _dive(1, priority="HIGH"),
            _dive(3, priority="LOW"),
        ]
    )
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 1


async def test_laser_preprocessing_skips_dives_with_every_image_labeled(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    # dive 1: every image has a laser label (mix of completed) -> excluded.
    # dive 2: one image still has no label at all -> picked.
    # dive 3: every image labeled in different projects -> excluded.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all(
        [
            _image(11, 1),
            _image(12, 1),
            _image(21, 2),
            _image(22, 2),
            _image(31, 3),
        ]
    )
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=True, label_studio_project_id=43),
            LaserLabel(image_id=12, completed=True, label_studio_project_id=72),
            LaserLabel(image_id=21, completed=True, label_studio_project_id=43),
            # image 22 has no laser_label -> dive 2 stays in cohort.
            LaserLabel(image_id=31, completed=True, label_studio_project_id=43),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 2


async def test_laser_preprocessing_treats_incomplete_label_as_labeled(session):
    """Once populate seeds an incomplete LaserLabel row, the dive must
    drop out of the cohort even though no labeler has touched it yet.

    This is the change that prevents the steady-state waste of re-
    staging raw bytes from NAS every hour for already-preprocessed
    dives. An incomplete sentinel row is what populate writes
    immediately after pushing an LS task — the JPEG already exists on
    the file-exchange and on NAS.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    # dive 1: image 11 has only an incomplete label -> dive excluded.
    # dive 2: image 21 has no label at all -> dive included.
    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=False, label_studio_project_id=99),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 2


async def test_laser_preprocessing_excludes_dive_with_only_incomplete_labels(session):
    """All images have only incomplete labels (with real project_id) —
    dive must drop out.

    This is the steady-state for the populate -> wait-for-labelers
    period. Without the predicate change, the dive would stay in the
    cohort and re-fire preprocess hourly until labelers completed.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=False, label_studio_project_id=99),
            LaserLabel(image_id=12, completed=False, label_studio_project_id=99),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


async def test_laser_preprocessing_excludes_dive_when_sentinel_coexists_with_real_label(
    session,
):
    """Defensive: an image carrying BOTH a NULL-project sentinel AND a
    real-project row should still drop the dive (the real row counts).
    Pinned because a future refactor that 'simplifies' the predicate
    to ignore project_id entirely would silently break this."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=False, label_studio_project_id=None
            ),
            LaserLabel(
                image_id=11, completed=False, label_studio_project_id=99
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


async def test_laser_preprocessing_ignores_null_project_sentinels(session):
    """NULL-`project_id` LaserLabel rows are legacy sentinels (~2000 in
    prod, one per HIGH-priority canonical image, source predates the
    Create-on-populate flow). The cohort selector must treat them as
    'no real label' so prod's existing sentinel population doesn't
    permanently drain the cohort. Mirrors the
    `project_id != None` filter the discovery endpoint at
    `label_controller.py:274` already has."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    # dive 1: image 11 has only a NULL-project sentinel -> dive INCLUDED.
    # dive 2: image 21 has a real-project incomplete row -> dive EXCLUDED.
    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=False, label_studio_project_id=None
            ),
            LaserLabel(
                image_id=21, completed=False, label_studio_project_id=99
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 1


async def test_laser_preprocessing_returns_none_when_no_unlabeled_images(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=True, label_studio_project_id=43),
            LaserLabel(image_id=21, completed=True, label_studio_project_id=43),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


async def test_laser_preprocessing_returns_none_with_no_high_priority(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    # LOW dives with unlabeled images are still excluded by the
    # priority filter.
    session.add_all([_dive(1, priority="LOW"), _dive(2, priority="LOW")])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


async def test_laser_discovery_and_selector_agree_on_real_label_definition(session):
    """Cross-controller consistency: the discovery endpoint
    (`get_laser_label_studio_project_ids`) and the cohort selector
    (`select_next_for_laser_preprocessing`) must use the same
    definition of 'real label' — `label_studio_project_id IS NOT
    NULL`. If a future refactor relaxes one but not the other, the
    cohort and the populate fan-out would disagree about which images
    are 'done' and either re-import duplicate LS tasks (selector too
    permissive) or leak unprocessed images (selector too strict).

    Concretely: a fixture where some images carry only sentinels and
    others carry real-project rows must produce a discovery list that
    excludes NULL and a selector pick that includes only the
    sentinel-only dives.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.controllers.label_controller import (  # pylint: disable=import-outside-toplevel
        get_laser_label_studio_project_ids,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            # dive 1 / image 11: only a sentinel -> dive 1 stays in
            # cohort. Sentinel must NOT show up in discovery.
            LaserLabel(
                image_id=11, completed=False, label_studio_project_id=None
            ),
            # dive 2 / image 21: real-project incomplete row -> dive 2
            # drops out of cohort. Project 99 must appear in discovery.
            LaserLabel(
                image_id=21, completed=False, label_studio_project_id=99
            ),
        ]
    )
    await session.flush()

    discovered = await get_laser_label_studio_project_ids(
        incomplete=True, session=session
    )
    selected = await select_next_for_laser_preprocessing(session=session)

    assert None not in discovered
    assert set(discovered) == {99}
    # Selector picks the dive whose only label is a sentinel — i.e.
    # the dive whose images have no real-project rows.
    assert selected == 1


async def test_laser_preprocessing_prod_state_sentinels_dont_drain_cohort(session):
    """Regression: 2026-05-03 prod incident. ~2000 NULL-`project_id`
    LaserLabel sentinel rows existed (one per HIGH-priority canonical
    image, source predates the Create-on-populate flow). The first
    cut of the new cohort predicate treated those rows as 'image is
    labeled' and the cohort drained to zero immediately on deploy.
    Pinning the prod-state shape here so that bug can't come back:
    multiple HIGH-priority dives, every image carrying ONLY a
    sentinel row, must still produce the lowest-id dive from the
    selector.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.laser_label import (  # pylint: disable=import-outside-toplevel
        LaserLabel,
    )

    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    image_ids: list[int] = []
    for dive_id in (1, 2, 3):
        for img_id in (dive_id * 10 + 1, dive_id * 10 + 2):
            session.add(_image(img_id, dive_id))
            image_ids.append(img_id)
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=img_id,
                completed=False,
                label_studio_project_id=None,
            )
            for img_id in image_ids
        ]
    )
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) == 1


async def test_laser_preprocessing_excludes_dive_with_no_images(session):
    """A dive with no images has no work for stage 0.1 — excluded."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_preprocessing,
    )

    session.add(_dive(1))
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


# ---------- stage 1: dive-frame-clustering ----------
#
# Cohort: HIGH + has at least one image carrying a *valid* LaserLabel
# (completed, not superseded, x/y both set) AND has zero PREDICTION
# DiveFrameCluster rows. Cascades from valid lasers like the
# headtail/species pipelines do; once a PREDICTION cluster exists the
# dive drops out (one-shot per dive).


async def test_clustering_requires_valid_laser_and_no_prediction_cluster(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_frame_clustering,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    # dive 1: valid laser + already has PREDICTION cluster -> excluded.
    # dive 2: valid laser + no clusters -> picked.
    # dive 3: no valid laser -> excluded.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=True, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=False, superseded=False,
                x=120.0, y=220.0, label_studio_project_id=43,
            ),
            DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION),
        ]
    )
    await session.flush()

    assert await select_next_for_dive_frame_clustering(session=session) == 2


async def test_clustering_excludes_dive_with_only_label_studio_cluster(session):
    """LABEL_STUDIO clusters come from stage 6.1 (label-time grouping)
    and don't count as PREDICTION clusters from stage 1. A dive whose
    only clusters are LABEL_STUDIO must still be picked for stage 1."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_frame_clustering,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.LABEL_STUDIO))
    await session.flush()

    assert await select_next_for_dive_frame_clustering(session=session) == 1


async def test_clustering_excludes_incomplete_or_superseded_or_null_xy_lasers(session):
    """Same gate as headtail: laser must be completed AND
    not superseded AND have both x and y populated to count as
    'valid laser', because that's what calibration and the validator
    treat as usable."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_frame_clustering,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _dive(2), _dive(3), _dive(4)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3), _image(41, 4)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=False, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=True, superseded=True,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=True, superseded=False,
                x=None, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=41, completed=True, superseded=False,
                x=100.0, y=None, label_studio_project_id=43,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_dive_frame_clustering(session=session) is None


async def test_clustering_returns_none_with_no_high_priority(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_frame_clustering,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, priority="LOW"))
    await session.flush()
    session.add(_image(11, 1))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    await session.flush()

    assert await select_next_for_dive_frame_clustering(session=session) is None


async def test_clustering_partial_persist_is_a_poison_pill(session):
    """`persist_dive_frame_clusters_activity` POSTs clusters one at a
    time. If a parent fails mid-persist (network blip, etc.), the
    dive is left with a partial PREDICTION cluster set covering some
    images but not others. The cohort selector excludes dives with
    *any* PREDICTION cluster, so this state is a poison pill — the
    next firing skips the dive forever even though the work isn't
    complete. An operator must drop the partial rows manually before
    the dive re-enters the cohort.

    Pinning this here so a future "smarter" predicate (e.g. "all
    images covered by some cluster") doesn't silently change the
    operational story without updating the activity docstring.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_frame_clustering,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    # Three laser-valid images, but only one PREDICTION cluster
    # covering one image — the other two would have been clustered on
    # a successful run.
    session.add_all([_image(11, 1), _image(12, 1), _image(13, 1)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=image_id, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            )
            for image_id in (11, 12, 13)
        ]
    )
    session.add(
        DiveFrameCluster(
            dive_id=1, data_source=DataSource.PREDICTION, image_ids=[11],
        )
    )
    await session.flush()

    # Even though images 12 and 13 are uncovered, the dive drops out
    # of the cohort. Operator action required to recover.
    assert await select_next_for_dive_frame_clustering(session=session) is None


# ---------- stage 2: species-preprocessing ----------
#
# Cohort flipped on 2026-05-05 from "any image without species label"
# → "any image carrying a valid laser label that lacks a non-sentinel
# species label." Cascades from valid lasers like the headtail
# pipeline does, but keeps the PREDICTION-cluster gate so the
# data-worker fan-out still gets the temporal-grouping context for
# the cluster overlay.


async def test_species_preprocessing_requires_prediction_cluster_and_valid_laser(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    # dive 1: no PREDICTION cluster -> excluded.
    # dive 2: PREDICTION cluster but no laser-valid image -> excluded.
    # dive 3: PREDICTION cluster + laser-valid image IN that cluster,
    #         without a species label -> picked.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            DiveFrameCluster(id=902, dive_id=2, data_source=DataSource.PREDICTION),
            DiveFrameCluster(id=903, dive_id=3, data_source=DataSource.PREDICTION),
            # The qualifying image must be IN the cluster, matching the resolver.
            DiveFrameClusterImageMapping(dive_frame_cluster_id=903, image_id=31),
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=False, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=True, superseded=False,
                x=120.0, y=220.0, label_studio_project_id=43,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) == 3


async def test_species_preprocessing_excludes_dive_with_only_incomplete_species_labels(
    session,
):
    """Once populate seeds an incomplete SpeciesLabel (with a real
    project_id) for every laser-valid image, the dive drops out of
    the cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION))
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=12, completed=True, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            SpeciesLabel(
                image_id=11, completed=False, label_studio_project_id=70
            ),
            SpeciesLabel(
                image_id=12, completed=False, label_studio_project_id=70
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) is None


async def test_species_preprocessing_excludes_dive_when_sentinel_coexists_with_real_label(
    session,
):
    """Defensive — see laser-stage analogue."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add_all(
        [
            SpeciesLabel(
                image_id=11, completed=False, label_studio_project_id=None
            ),
            SpeciesLabel(
                image_id=11, completed=False, label_studio_project_id=70
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) is None


async def test_species_preprocessing_ignores_null_project_species_sentinels(session):
    """NULL-`project_id` rows are legacy sentinels — see the laser
    cohort docstring. They must NOT drop a dive from the species
    cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameClusterImageMapping,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(DiveFrameCluster(id=910, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=910, image_id=11))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add(
        SpeciesLabel(
            image_id=11, completed=False, label_studio_project_id=None
        ),
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) == 1


async def test_species_preprocessing_excludes_incomplete_or_superseded_or_null_xy_lasers(
    session,
):
    """Same gate as headtail — the species cohort cascades from the
    same 'valid laser' definition."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _dive(2), _dive(3), _dive(4)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3), _image(41, 4)])
    await session.flush()
    session.add_all(
        [
            DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION),
            DiveFrameCluster(dive_id=2, data_source=DataSource.PREDICTION),
            DiveFrameCluster(dive_id=3, data_source=DataSource.PREDICTION),
            DiveFrameCluster(dive_id=4, data_source=DataSource.PREDICTION),
            LaserLabel(
                image_id=11, completed=False, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=True, superseded=True,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=True, superseded=False,
                x=None, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=41, completed=True, superseded=False,
                x=100.0, y=None, label_studio_project_id=43,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) is None


async def test_species_preprocessing_returns_none_when_only_label_studio_clusters(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.LABEL_STUDIO))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) is None


# ---------- species population (scheduled populate parent cohort) ----------
#
# Superseded-aware, PREDICTION-cluster-free, returns ALL matching dives.
# The distinguishing behaviour vs species-preprocessing: a dive whose
# species rows are all superseded (post hosted-LS migration) re-enters
# the cohort, and a live (non-superseded, real-project) row excludes it.


async def test_needing_species_population_picks_laser_valid_without_live_species(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_dives_needing_species_population,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _dive(2, priority="LOW")])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            # dive 1: HIGH + laser-valid + no species row -> picked. No
            # PREDICTION cluster required (unlike species-preprocessing).
            LaserLabel(
                image_id=11, completed=True, superseded=False, x=1.0, y=2.0,
                label_studio_project_id=43,
            ),
            # dive 2: LOW priority -> excluded.
            LaserLabel(
                image_id=21, completed=True, superseded=False, x=1.0, y=2.0,
                label_studio_project_id=43,
            ),
        ]
    )
    await session.flush()

    assert await select_dives_needing_species_population(session=session) == [1]


async def test_needing_species_population_reincludes_superseded_only_dive(session):
    """The migration case: a dive whose only species rows are superseded
    (old dead project) must re-enter BOTH cohorts. species-preprocessing
    used to exclude it (its check ignored `superseded`); the two
    selectors disagreeing deadlocked the stage — fixed 2026-07-21."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_dives_needing_species_population,
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(DiveFrameCluster(id=950, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    # image 11 is IN the cluster — the preprocessing selector requires this.
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=950, image_id=11))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False, x=1.0, y=2.0,
            label_studio_project_id=43,
        )
    )
    # only a superseded species row (dead old project 117)
    session.add(
        SpeciesLabel(
            image_id=11, completed=False, superseded=True,
            label_studio_project_id=117,
        )
    )
    await session.flush()

    assert await select_dives_needing_species_population(session=session) == [1]
    # The preprocessing selector MUST agree. It used to return None here
    # (its check ignored `superseded`), and that disagreement was a
    # deadlock: populate kept re-selecting the dive but every image was
    # deferred on a JPEG that preprocess would never regenerate, so
    # `deferred > 0` and the per-dive project never published.
    assert await select_next_for_species_preprocessing(session=session) == 1


async def test_needing_species_population_excludes_dive_with_live_species(session):
    """A non-superseded, real-project species row is a live task -> the
    image no longer needs population, so the dive drops out."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_dives_needing_species_population,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False, x=1.0, y=2.0,
            label_studio_project_id=43,
        )
    )
    session.add(
        SpeciesLabel(
            image_id=11, completed=False, superseded=False,
            label_studio_project_id=226,
        )
    )
    await session.flush()

    assert await select_dives_needing_species_population(session=session) == []


async def test_needing_species_population_excludes_invalid_laser(session):
    """Same valid-laser gate as the other cascade cohorts."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_dives_needing_species_population,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1), _image(13, 1)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=False, superseded=False, x=1.0, y=2.0),
            LaserLabel(image_id=12, completed=True, superseded=True, x=1.0, y=2.0),
            LaserLabel(image_id=13, completed=True, superseded=False, x=None, y=2.0),
        ]
    )
    await session.flush()

    assert await select_dives_needing_species_population(session=session) == []


# ---------- stage 5.1: headtail-preprocessing ----------
#
# Cohort cascades from valid laser labels: any image with a "valid"
# laser label (completed=True, superseded=False, x/y both set —
# matches what perform_laser_calibration and validate_laser_labels
# treat as usable) drops into the headtail pipeline as soon as it
# lacks a non-sentinel HeadTailLabel row. Species labeling is no
# longer in the path; head/tail can run in parallel with stages 1/2/4.


async def test_headtail_preprocessing_requires_valid_laser_without_any_headtail(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    # dive 1: valid laser + headtail row exists -> excluded.
    # dive 2: valid laser, no headtail row -> picked.
    # dive 3: laser row is null x/y (no-laser sentinel) -> excluded.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=True, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=True, superseded=False,
                x=None, y=None, label_studio_project_id=43,
            ),
        ]
    )
    session.add(
        HeadTailLabel(image_id=11, completed=True, label_studio_project_id=71)
    )
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) == 2


async def test_headtail_preprocessing_excludes_dive_with_only_incomplete_headtail(
    session,
):
    """Once populate seeds an incomplete HeadTailLabel (with a real
    project_id) for every laser-cascaded image, the dive drops out of
    the cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add(
        HeadTailLabel(
            image_id=11, completed=False, label_studio_project_id=71
        )
    )
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) is None


async def test_headtail_preprocessing_excludes_dive_when_sentinel_coexists_with_real_label(
    session,
):
    """Defensive — see laser-stage analogue."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add_all(
        [
            HeadTailLabel(
                image_id=11, completed=False, label_studio_project_id=None
            ),
            HeadTailLabel(
                image_id=11, completed=False, label_studio_project_id=71
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) is None


async def test_headtail_preprocessing_ignores_null_project_sentinels(session):
    """NULL-`project_id` HeadTailLabel rows are legacy sentinels — see
    the dive-image cohort docstring. They must NOT drop a dive from
    the headtail cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add(
        HeadTailLabel(
            image_id=11, completed=False, label_studio_project_id=None
        )
    )
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) == 1


async def test_headtail_preprocessing_excludes_incomplete_or_superseded_or_null_xy_lasers(
    session,
):
    """A laser label only counts if completed AND not superseded AND
    has both x and y populated. Anything weaker is not a "valid
    label" per the gate."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    # dive 1: laser is not completed.
    # dive 2: laser is superseded.
    # dive 3: laser is missing x.
    # dive 4: laser is missing y.
    session.add_all([_dive(1), _dive(2), _dive(3), _dive(4)])
    await session.flush()
    session.add_all(
        [_image(11, 1), _image(21, 2), _image(31, 3), _image(41, 4)]
    )
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=False, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=True, superseded=True,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=True, superseded=False,
                x=None, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=41, completed=True, superseded=False,
                x=100.0, y=None, label_studio_project_id=43,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) is None


async def test_headtail_preprocessing_returns_none_when_no_laser_labels(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) is None


# ---------- stage 9: slate-preprocessing ----------


async def test_slate_preprocessing_requires_dive_slate_id_and_marker(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    # dive 1: HIGH but no dive_slate_id -> excluded.
    # dive 2: HIGH + dive_slate_id but no slate-marked species_label -> excluded.
    # dive 3: HIGH + dive_slate_id + slate-marked + no completed slate label -> picked.
    session.add_all(
        [
            _dive(1, dive_slate_id=None),
            _dive(2, dive_slate_id=99),
            _dive(3, dive_slate_id=99),
        ]
    )
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER),
            SpeciesLabel(image_id=21, content_of_image="Fish"),
            SpeciesLabel(image_id=31, content_of_image=SLATE_CONTENT_MARKER),
        ]
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) == 3


async def test_slate_preprocessing_skips_when_every_slate_image_labeled(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1, dive_slate_id=99), _dive(2, dive_slate_id=99)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER),
            SpeciesLabel(image_id=21, content_of_image=SLATE_CONTENT_MARKER),
            DiveSlateLabel(
                image_id=11, completed=True, label_studio_project_id=66
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) == 2


async def test_slate_preprocessing_excludes_dive_with_only_incomplete_slate_labels(
    session,
):
    """Once populate seeds an incomplete DiveSlateLabel (with a real
    project_id) for every slate-marked image, the dive drops out of
    the cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=99))
    await session.flush()
    session.add(_image(11, 1))
    session.add(SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER))
    session.add(
        DiveSlateLabel(
            image_id=11, completed=False, label_studio_project_id=66
        )
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) is None


async def test_slate_preprocessing_excludes_dive_when_sentinel_coexists_with_real_label(
    session,
):
    """Defensive — see laser-stage analogue."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=99))
    await session.flush()
    session.add(_image(11, 1))
    session.add(SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER))
    session.add_all(
        [
            DiveSlateLabel(
                image_id=11, completed=False, label_studio_project_id=None
            ),
            DiveSlateLabel(
                image_id=11, completed=False, label_studio_project_id=66
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) is None


async def test_slate_preprocessing_ignores_null_project_sentinels(session):
    """NULL-`project_id` DiveSlateLabel rows are legacy sentinels — see
    the dive-image cohort docstring. They must NOT drop a dive from
    the slate cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        SLATE_CONTENT_MARKER,
        select_next_for_slate_preprocessing,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=99))
    await session.flush()
    session.add(_image(11, 1))
    session.add(SpeciesLabel(image_id=11, content_of_image=SLATE_CONTENT_MARKER))
    session.add(
        DiveSlateLabel(
            image_id=11, completed=False, label_studio_project_id=None
        )
    )
    await session.flush()

    assert await select_next_for_slate_preprocessing(session=session) == 1


# ---------- stage 13: laser-calibration ----------


async def test_laser_calibration_requires_min_completed_slate_labels(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_calibration,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    # dive 1: 1 completed slate label -> below threshold, excluded.
    # dive 2: 2 completed slate labels -> picked.
    # dive 3: 3 completed slate labels but already has extrinsics -> excluded.
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add_all(
        [
            _dive(1, dive_slate_id=99),
            _dive(2, dive_slate_id=99),
            _dive(3, dive_slate_id=99),
        ]
    )
    await session.flush()
    session.add_all(
        [
            _image(11, 1),
            _image(21, 2),
            _image(22, 2),
            _image(31, 3),
            _image(32, 3),
            _image(33, 3),
        ]
    )
    await session.flush()
    session.add_all(
        [
            DiveSlateLabel(image_id=11, completed=True),
            DiveSlateLabel(image_id=21, completed=True),
            DiveSlateLabel(image_id=22, completed=True),
            DiveSlateLabel(image_id=31, completed=True),
            DiveSlateLabel(image_id=32, completed=True),
            DiveSlateLabel(image_id=33, completed=True),
            LaserExtrinsics(dive_id=3, camera_id=1),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_calibration(session=session) == 2


async def test_laser_calibration_requires_dive_slate_id(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_laser_calibration,
    )
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=None))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    session.add_all(
        [
            DiveSlateLabel(image_id=11, completed=True),
            DiveSlateLabel(image_id=12, completed=True),
        ]
    )
    await session.flush()

    assert await select_next_for_laser_calibration(session=session) is None


# ---------- stage 14: measure-fish ----------
#
# Cohort mirrors dive_pipeline_status.measured: a dive is selected iff it
# has extrinsics and at least one *measurable* image with no measurement.
# It used to key on "has a LABEL_STUDIO cluster with fish_id IS NULL",
# which never goes false — a cluster is only bound through a measurable
# image, so clusters without one kept every dive in the cohort forever.
# Prod dive 466 carried 1632 such clusters against 24 measurable images.


MEASURABLE_CONTENT = "Fish, Hogfish (Lachnolaimus maximus)"


def _measurable_image(
    session,
    image_id: int,
    dive_id: int,
    *,
    cluster_id: int,
    content_of_image: str | None = MEASURABLE_CONTENT,
):
    """An image stage 14 would attempt: top-three species label + valid
    laser + valid headtail + a LABEL_STUDIO cluster.

    `content_of_image` defaults to a real `Fish` row because stage 14 also
    needs a `Common (Scientific)` name to measure against — a row without
    one is skipped by the activity, so leaving it NULL here would have
    built an image the pipeline can never actually measure.
    """
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_image(image_id, dive_id))
    session.add(
        DiveFrameCluster(
            id=cluster_id, dive_id=dive_id, data_source=DataSource.LABEL_STUDIO
        )
    )
    session.add(
        LaserLabel(image_id=image_id, completed=True, superseded=False, x=1.0, y=2.0)
    )
    session.add(
        HeadTailLabel(
            image_id=image_id, completed=True, superseded=False,
            head_x=1.0, head_y=2.0, tail_x=3.0, tail_y=4.0,
        )
    )
    session.add(
        SpeciesLabel(
            image_id=image_id, top_three_photos_of_group=True,
            completed=True, superseded=False, label_studio_project_id=70,
            content_of_image=content_of_image,
        )
    )
    return DiveFrameClusterImageMapping(
        dive_frame_cluster_id=cluster_id, image_id=image_id
    )


def _measurement(image_id: int, fish_id: int = 100):
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    return Measurement(image_id=image_id, fish_id=fish_id, length_m=0.3)


async def test_measure_fish_requires_extrinsics_and_an_unmeasured_measurable_image(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    # dive 1: measurable image but no extrinsics -> excluded.
    # dive 2: extrinsics + measurable image already measured -> excluded.
    # dive 3: extrinsics + unmeasured measurable image -> picked.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all(
        [LaserExtrinsics(dive_id=2, camera_id=1), LaserExtrinsics(dive_id=3, camera_id=1)]
    )
    m1 = _measurable_image(session, 11, 1, cluster_id=1)
    m2 = _measurable_image(session, 21, 2, cluster_id=2)
    m3 = _measurable_image(session, 31, 3, cluster_id=3)
    await session.flush()
    session.add_all([m1, m2, m3])
    session.add(_measurement(21))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 3


async def test_measure_fish_returns_none_when_everything_measurable_is_measured(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add(_measurement(11))
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None


async def test_measure_fish_ignores_unbound_clusters_with_no_measurable_image(session):
    """The regression: an unbound cluster stage 14 can never touch must
    not keep the dive in the cohort forever."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add(_measurement(11))
    session.add(
        DiveFrameCluster(
            id=99, dive_id=1, data_source=DataSource.LABEL_STUDIO, fish_id=None
        )
    )
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None


async def test_measure_fish_selects_dive_with_borrowed_calibration(session):
    """A fish dive with no extrinsics of its own is still measurable when it
    links to a slate dive that owns extrinsics via `calibration_dive_id`."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    # dive 1: the slate/calibration dive — owns extrinsics, no fish images.
    # dive 2: the fish dive — borrows dive 1's calibration, has an
    # unmeasured measurable image. Without the link it would be excluded
    # (no extrinsics); with it, it is selected.
    session.add_all([_dive(1), _dive(2, calibration_dive_id=1)])
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    mapping = _measurable_image(session, 21, 2, cluster_id=2)
    await session.flush()
    session.add(mapping)
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 2


async def test_measure_fish_link_to_uncalibrated_source_does_not_select(session):
    """A link to a source that owns no extrinsics doesn't fabricate
    calibration — the fish dive stays out of the cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )

    session.add_all([_dive(1), _dive(2, calibration_dive_id=1)])
    await session.flush()
    mapping = _measurable_image(session, 21, 2, cluster_id=2)
    await session.flush()
    session.add(mapping)
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None


# ── Superseded rows must not permanently strand an image ──────────────
#
# The breach-recovery pass dead-lettered ~1.9k species and ~1.9k headtail
# rows that pointed at retired old-LS projects. `needing-species-population`
# is superseded-aware, so those dives re-enter *populate* — but the
# preprocess selectors' existence gates were not, so the same images read as
# "already labeled" and never got their stage-2 JPEG regenerated. Populate
# then defers them forever (`deferred > 0`), so the per-dive project never
# publishes. Measured in prod on 2026-07-21: 1,826 species and 1,761 headtail
# images blocked this way, across the 12 dives that own every species project.
#
# A superseded row is a dead letter, not evidence the work is done.


async def test_species_preprocessing_reenters_dive_with_only_superseded_species(
    session,
):
    """A superseded real-project SpeciesLabel must NOT gate the image out."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameClusterImageMapping,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(DiveFrameCluster(id=920, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=920, image_id=11))
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            # Dead-lettered row pointing at a retired project.
            SpeciesLabel(
                image_id=11, completed=False, superseded=True,
                label_studio_project_id=117,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) == 1


async def test_headtail_preprocessing_reenters_dive_with_only_superseded_headtail(
    session,
):
    """Same dead-letter rule for the head/tail cohort."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_headtail_preprocessing,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            HeadTailLabel(
                image_id=11, completed=False, superseded=True,
                label_studio_project_id=71,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_headtail_preprocessing(session=session) == 1


async def test_species_preprocessing_still_excludes_live_real_species_row(session):
    """Guard the other direction: a LIVE (non-superseded) real-project row
    must still drop the dive, or preprocess would loop forever."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION))
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            SpeciesLabel(
                image_id=11, completed=False, superseded=False,
                label_studio_project_id=70,
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) is None


# ── Stage 14 only measures rows that carry a scientific name ──────────
#
# `measure_fish_activity._parse_species_names` needs the trailing
# ", "-chunk of `content_of_image` to look like "Common (Scientific)".
# The non-`Fish` taxonomy branches don't:
#
#     "Fish Model, Weasly Fish"     -> skipped
#     "Calibration Targets, Ruler"  -> skipped
#
# If the cohort ignores that, it offers an image the activity always skips.
# No Measurement is ever written, so `~is_measured` stays true and the dive
# is re-selected every hour forever — the same never-goes-false shape that
# blocked scheduling stage 14 before 2026-07-17.


async def test_measure_fish_skips_species_rows_without_a_scientific_name(session):
    """Fish Model / Calibration Targets rows must not hold a dive in the
    cohort — nothing downstream can ever measure them."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    for content in ("Fish Model, Weasly Fish", "Calibration Targets, Ruler", None):
        session.add(_dive(1))
        await session.flush()
        session.add(LaserExtrinsics(dive_id=1, camera_id=1))
        mapping = _measurable_image(
            session, 11, 1, cluster_id=1, content_of_image=content
        )
        await session.flush()
        session.add(mapping)
        await session.flush()

        assert await select_next_for_measure_fish(session=session) is None, content

        await session.rollback()


async def test_measure_fish_still_selects_a_real_fish_row(session):
    """Guard the other direction — the `Fish` branch stays measurable."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_measure_fish,
    )
    from fishsense_api.models.laser_extrinsics import (  # pylint: disable=import-outside-toplevel
        LaserExtrinsics,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    mapping = _measurable_image(session, 11, 1, cluster_id=1)  # default Fish row
    await session.flush()
    session.add(mapping)
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 1


async def test_species_preprocessing_skips_qualifying_image_not_in_a_cluster(session):
    """The 2026-07-22 poison pill.

    A dive can have PREDICTION clusters AND a valid-laser image with no
    species row, yet that image not be in any cluster — e.g. its laser was
    validated after stage-1 clustering (one-shot per dive). The selector used
    to check the two conditions dive-wide and independently, so it picked such
    a dive; the resolver, which only dispatches per-image work for images IN a
    cluster, then returned zero. Ordered by id and drained one dive per hour,
    that dive sat at the front forever and starved every productive dive
    behind it (dives 59 and 439 in prod).

    So a qualifying image that is NOT clustered must not select the dive.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    # image 11 is clustered but already handled; image 12 qualifies but is NOT
    # in any cluster — the exact poison-pill shape.
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(DiveFrameCluster(id=930, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=930, image_id=11))
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=1.0, y=2.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=12, completed=True, superseded=False,
                x=3.0, y=4.0, label_studio_project_id=43,
            ),
        ]
    )
    await session.flush()

    # image 11 (clustered) has no species row either, so on its own it WOULD
    # select the dive; give it one so the only *unlabeled* image is the
    # unclustered 12. Now nothing processable remains.
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(
        SpeciesLabel(image_id=11, completed=False, label_studio_project_id=70)
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) is None


async def test_species_preprocessing_picks_dive_with_a_clustered_qualifying_image(
    session,
):
    """Guard the other direction: when the qualifying image IS clustered, the
    dive is still selected — the fix must not over-exclude."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(DiveFrameCluster(id=940, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=940, image_id=11))
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=1.0, y=2.0, label_studio_project_id=43,
        )
    )
    await session.flush()

    assert await select_next_for_species_preprocessing(session=session) == 1

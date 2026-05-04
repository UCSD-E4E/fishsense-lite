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


def _dive(dive_id: int, *, priority="HIGH", dive_slate_id=None):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority[priority],
        dive_slate_id=dive_slate_id,
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


# ---------- stage 2: dive-image-preprocessing ----------


async def test_dive_image_preprocessing_requires_prediction_cluster(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )

    # dive 1: no clusters -> excluded.
    # dive 2: PREDICTION cluster + image without species_label -> picked.
    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(101, 1), _image(202, 2)])
    await session.flush()
    session.add(DiveFrameCluster(dive_id=2, data_source=DataSource.PREDICTION))
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) == 2


async def test_dive_image_preprocessing_skips_when_every_image_labeled(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    # dive 1: 1 image, all labeled -> excluded.
    # dive 2: 2 images, only one labeled -> picked.
    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(22, 2)])
    await session.flush()
    session.add_all(
        [
            DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION),
            DiveFrameCluster(dive_id=2, data_source=DataSource.PREDICTION),
        ]
    )
    session.add_all(
        [
            SpeciesLabel(image_id=11, completed=True, label_studio_project_id=70),
            SpeciesLabel(image_id=21, completed=True, label_studio_project_id=70),
            # image 22 has no species_label -> dive 2 stays in cohort.
        ]
    )
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) == 2


async def test_dive_image_preprocessing_excludes_dive_with_only_incomplete_labels(
    session,
):
    """Once populate seeds an incomplete SpeciesLabel (with a real
    project_id) for every image, the dive must drop from the cohort
    even though no labeler has completed it yet."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION))
    session.add_all(
        [
            SpeciesLabel(
                image_id=11, completed=False, label_studio_project_id=70
            ),
            SpeciesLabel(
                image_id=12, completed=False, label_studio_project_id=70
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) is None


async def test_dive_image_preprocessing_excludes_dive_when_sentinel_coexists_with_real_label(
    session,
):
    """Defensive — see laser-stage analogue."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION))
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

    assert await select_next_for_dive_image_preprocessing(session=session) is None


async def test_dive_image_preprocessing_ignores_null_project_sentinels(session):
    """NULL-`project_id` rows are legacy sentinels (~2000 in prod, one
    per HIGH-priority canonical image, source predates the
    Create-on-populate flow). The cohort selector must treat them as
    'no real label' so prod's existing sentinel population doesn't
    permanently drain the cohort. Mirrors the
    `project_id != None` filter every discovery endpoint already has.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.PREDICTION))
    session.add(
        SpeciesLabel(
            image_id=11, completed=False, label_studio_project_id=None
        ),
    )
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) == 1


async def test_dive_image_preprocessing_returns_none_when_only_label_studio_clusters(
    session,
):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_dive_image_preprocessing,
    )
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
    )

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    session.add(DiveFrameCluster(dive_id=1, data_source=DataSource.LABEL_STUDIO))
    await session.flush()

    assert await select_next_for_dive_image_preprocessing(session=session) is None


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


async def test_measure_fish_requires_extrinsics_and_unbound_label_studio_cluster(
    session,
):
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

    # dive 1: no extrinsics -> excluded.
    # dive 2: extrinsics + LABEL_STUDIO cluster all bound -> excluded.
    # dive 3: extrinsics + unbound LABEL_STUDIO cluster -> picked.
    # dive 4: extrinsics + only PREDICTION clusters -> excluded.
    session.add_all([_dive(1), _dive(2), _dive(3), _dive(4)])
    await session.flush()
    session.add_all(
        [
            LaserExtrinsics(dive_id=2, camera_id=1),
            LaserExtrinsics(dive_id=3, camera_id=1),
            LaserExtrinsics(dive_id=4, camera_id=1),
            DiveFrameCluster(
                dive_id=2, data_source=DataSource.LABEL_STUDIO, fish_id=42
            ),
            DiveFrameCluster(
                dive_id=3, data_source=DataSource.LABEL_STUDIO, fish_id=None
            ),
            DiveFrameCluster(
                dive_id=4, data_source=DataSource.PREDICTION, fish_id=None
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_measure_fish(session=session) == 3


async def test_measure_fish_returns_none_when_all_clusters_bound(session):
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
    session.add_all(
        [
            LaserExtrinsics(dive_id=1, camera_id=1),
            DiveFrameCluster(
                dive_id=1, data_source=DataSource.LABEL_STUDIO, fish_id=42
            ),
        ]
    )
    await session.flush()

    assert await select_next_for_measure_fish(session=session) is None

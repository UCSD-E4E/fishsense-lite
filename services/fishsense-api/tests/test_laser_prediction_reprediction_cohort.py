"""Re-predicting a dive whose detector version has moved on.

`select_next_for_laser_prediction` keys on the *absence* of a
`LaserPrediction`, so a single row — even a non-detection — removed that image
from the cohort permanently, and its own docstring called re-prediction "a
manual affair". Improving the stage therefore stranded every prediction already
written: prod carries 1,555 of them, 183 of which sit outside the
expected-laser region the stage now enforces.

The fix is the same shape `LaserDepth` and `Measurement` already use — select
on a *mismatch* with the version that produced the row rather than on absence.

Two deliberate limits on that:

* **Only dives still being labeled.** A version bump would otherwise re-predict
  the whole corpus, which is GPU time and NAS staging for dives nobody is
  looking at. A dive qualifies while it has at least one incomplete,
  non-superseded laser label — i.e. a labeler still has open tasks on it.
* **Never an image a human has finished.** A completed non-superseded
  `LaserLabel` takes the image out regardless of version, so re-prediction
  cannot disturb work already done. That is the guard, not a side effect.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

CURRENT = 2


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  pylint: disable=unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    await engine.dispose()


async def _dive(
    session,
    dive_id,
    *,
    version,
    completed_label=False,
    open_label=True,
    canonical=True,
):
    """One HIGH dive, one image, one prediction at `version`."""
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.laser_prediction import LaserPrediction
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 8, 1, tzinfo=timezone.utc)
    session.add(
        Dive(id=dive_id, path=f"/d{dive_id}", priority=Priority.HIGH, dive_datetime=when)
    )
    image_id = dive_id * 100
    session.add(
        Image(
            id=image_id,
            dive_id=dive_id,
            path=f"/d{dive_id}/a.ORF",
            taken_datetime=when,
            checksum=f"c{dive_id}",
            is_canonical=canonical,
        )
    )
    session.add(
        LaserPrediction(image_id=image_id, confidence=0.9, predictor_version=version)
    )
    if completed_label:
        session.add(
            LaserLabel(
                image_id=image_id,
                label_studio_project_id=1,
                completed=True,
                superseded=False,
            )
        )
    if open_label:
        # A second image carrying the dive's open task, so "still being
        # labeled" is true independently of the image under test.
        session.add(
            Image(
                id=image_id + 1,
                dive_id=dive_id,
                path=f"/d{dive_id}/b.ORF",
                taken_datetime=when,
                checksum=f"c{dive_id}b",
                is_canonical=True,
            )
        )
        session.add(
            LaserPrediction(
                image_id=image_id + 1, confidence=0.9, predictor_version=CURRENT
            )
        )
        session.add(
            LaserLabel(
                image_id=image_id + 1,
                label_studio_project_id=1,
                completed=False,
                superseded=False,
            )
        )
    await session.commit()
    return image_id


async def _select(session):
    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_laser_prediction,
    )

    return await select_next_for_laser_prediction(session=session)


async def test_stale_prediction_on_an_actively_labeled_dive_is_selected(session):
    await _dive(session, 1, version=CURRENT - 1)
    assert await _select(session) == 1


async def test_null_version_counts_as_stale(session):
    """The 1,555 rows already in prod predate versioning. NULL is 'unknown,
    therefore stale' — it must not read as current under three-valued logic."""
    await _dive(session, 1, version=None)
    assert await _select(session) == 1


async def test_current_version_is_not_reselected(session):
    """The cohort has to drain, or the dive re-fires every hour forever."""
    await _dive(session, 1, version=CURRENT)
    assert await _select(session) is None


async def test_stale_prediction_on_a_finished_dive_is_left_alone(session):
    """No open tasks: nobody is looking at this dive, so a bump must not spend
    GPU time and NAS staging on it."""
    await _dive(session, 1, version=CURRENT - 1, open_label=False)
    assert await _select(session) is None


async def test_a_humans_completed_label_protects_its_image(session):
    """Re-prediction must never disturb finished work.

    The dive is still being labeled — its second image has an open task — so
    the dive-level gate is satisfied. But the only image carrying a *stale*
    prediction is one a human has finished, so there is nothing to re-predict
    and the dive must not be selected. Without the completed-label guard this
    dive would be picked every hour and the labeler's own point overwritten.
    """
    await _dive(session, 1, version=CURRENT - 1, completed_label=True)
    assert await _select(session) is None


async def test_non_canonical_images_are_ignored(session):
    await _dive(session, 1, version=CURRENT - 1, canonical=False, open_label=False)
    assert await _select(session) is None


async def test_unpredicted_images_still_select_as_before(session):
    """The original predicate is unchanged: an image with no prediction at all
    is selected whether or not anyone is labeling the dive."""
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.image import Image
    from fishsense_api.models.priority import Priority

    when = datetime(2026, 8, 1, tzinfo=timezone.utc)
    session.add(Dive(id=5, path="/d5", priority=Priority.HIGH, dive_datetime=when))
    session.add(
        Image(
            id=500,
            dive_id=5,
            path="/d5/a.ORF",
            taken_datetime=when,
            checksum="c5",
            is_canonical=True,
        )
    )
    await session.commit()
    assert await _select(session) == 5

"""`GET /api/v1/labels/laser/label-studio-project-ids?gated=` filters on
whether the auto-accept gate has *finished* with a project's frames.

The landing page and the triage clients use this to avoid sending a human
at work the machine is about to do: an ungated frame still has a pending
auto-accept decision, so a labeler who judges it duplicates the gate.

The predicate is deliberately "the gate is done here", not "the gate has
run here". A dive the gate has swept halfway passes the weaker test while
still holding frames it is about to take — which is exactly the case the
filter exists to prevent. So `gated=true` requires *no* laser prediction
awaiting judgment, and at least one that has been judged; a project the
gate has never touched is not "gated" by vacuous truth, matching the
`*_labeling_complete` convention in `dive_pipeline_status`.

Note this makes re-prediction self-correcting: the persist activity
clears `gate_verdict` on upsert, so a re-predicted dive drops off the
landing page until the gate has been back through it.

In-memory-sqlite fixture style, matching
test_label_studio_project_ids_superseded.py — testing query composition,
not referential integrity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


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


def _image(image_id: int, dive_id: int = 1):
    from fishsense_api.models.image import Image

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"{image_id:032d}",
        dive_id=dive_id,
    )


def _laser_label(label_id, image_id, *, project_id, completed=True, superseded=False):
    from fishsense_api.models.laser_label import LaserLabel

    return LaserLabel(
        id=label_id,
        image_id=image_id,
        completed=completed,
        superseded=superseded,
        label_studio_project_id=project_id,
    )


def _prediction(prediction_id, image_id, *, gate_verdict):
    from fishsense_api.models.laser_prediction import LaserPrediction

    return LaserPrediction(
        id=prediction_id,
        image_id=image_id,
        confidence=0.9,
        gate_verdict=gate_verdict,
    )


async def _ids(session, **kwargs):
    from fishsense_api.controllers.label_controller import (
        get_laser_label_studio_project_ids,
    )

    return sorted(await get_laser_label_studio_project_ids(session=session, **kwargs))


async def _seed(session, rows):
    for row in rows:
        session.add(row)
    await session.flush()


# Project 10: both frames judged by the gate -> the gate is done here.
# Project 20: one frame judged, one still pending -> the gate is mid-sweep.
# Project 30: frames exist but the gate has never run.
FULLY_GATED, PARTIALLY_GATED, UNGATED = 10, 20, 30


async def _seed_three_projects(session):
    await _seed(
        session,
        [
            _image(1),
            _image(2),
            _image(3),
            _image(4),
            _image(5),
            _laser_label(1, 1, project_id=FULLY_GATED),
            _laser_label(2, 2, project_id=FULLY_GATED),
            _laser_label(3, 3, project_id=PARTIALLY_GATED),
            _laser_label(4, 4, project_id=PARTIALLY_GATED),
            _laser_label(5, 5, project_id=UNGATED),
            _prediction(1, 1, gate_verdict="auto_accepted"),
            _prediction(2, 2, gate_verdict="off_line"),
            _prediction(3, 3, gate_verdict="auto_accepted"),
            _prediction(4, 4, gate_verdict=None),
            _prediction(5, 5, gate_verdict=None),
        ],
    )


async def test_omitted_gated_is_unchanged(session):
    """Default behaviour must not move — every project still surfaces."""
    await _seed_three_projects(session)
    assert await _ids(session) == [FULLY_GATED, PARTIALLY_GATED, UNGATED]


async def test_gated_true_requires_the_gate_to_be_finished(session):
    """A half-swept project is excluded — its pending frames are the
    machine's work, not a labeler's."""
    await _seed_three_projects(session)
    assert await _ids(session, gated=True) == [FULLY_GATED]


async def test_gated_true_excludes_a_project_the_gate_never_touched(session):
    """Vacuous truth reads as False, per the `dive_pipeline_status`
    convention: no judged prediction means the gate has not been here."""
    await _seed_three_projects(session)
    assert UNGATED not in await _ids(session, gated=True)


async def test_gated_false_is_the_exact_complement(session):
    """`gated=false` must partition the same base set, so the two answers
    reassemble into the unfiltered one. Anything else makes the flag a
    third, silently different query."""
    await _seed_three_projects(session)
    unfiltered = await _ids(session)
    assert sorted(
        await _ids(session, gated=True) + await _ids(session, gated=False)
    ) == unfiltered


async def test_a_project_with_no_predictions_at_all_is_not_gated(session):
    """No prediction row means the detector has not run, so there is
    nothing for the gate to have judged."""
    await _seed(
        session,
        [_image(1), _laser_label(1, 1, project_id=FULLY_GATED)],
    )
    assert not await _ids(session, gated=True)
    assert await _ids(session, gated=False) == [FULLY_GATED]


async def test_superseded_labels_do_not_hold_a_project_ungated(session):
    """A dead-lettered label is not live labeling work, so its image's
    pending prediction must not keep an otherwise-finished project off the
    landing page. Mirrors the `superseded == False` filter every other
    laser read applies."""
    await _seed(
        session,
        [
            _image(1),
            _image(2),
            _laser_label(1, 1, project_id=FULLY_GATED),
            _laser_label(2, 2, project_id=FULLY_GATED, superseded=True),
            _prediction(1, 1, gate_verdict="auto_accepted"),
            _prediction(2, 2, gate_verdict=None),
        ],
    )
    assert await _ids(session, gated=True) == [FULLY_GATED]


async def test_gated_composes_with_incomplete(session):
    """Both filters apply together — the landing page sends `incomplete`
    and `gated` in the same request."""
    await _seed(
        session,
        [
            _image(1),
            _image(2),
            # Gate finished, but every label is already done: no work left.
            _laser_label(1, 1, project_id=FULLY_GATED, completed=True),
            # Gate finished and work remains -> this is the one to show.
            _laser_label(2, 2, project_id=PARTIALLY_GATED, completed=False),
            _prediction(1, 1, gate_verdict="auto_accepted"),
            _prediction(2, 2, gate_verdict="audit_sample"),
        ],
    )
    assert await _ids(session, gated=True, incomplete=True) == [PARTIALLY_GATED]

"""`Priority.NONE` + `Dive.notes` — parking a dive with a stated reason.

Some dives can never complete the pipeline for a reason that lives outside
the data: dive 437 was labelled `V-Slate 7`, and no scan of that slate
exists, so it can never be calibrated and can never be measured. LOW was
the only way to keep it out of the hourly cohorts, but LOW means "not yet"
— it is the state every freshly-ingested dive passes through — so a parked
dive is indistinguishable from one merely waiting its turn, and the reason
lives nowhere at all.

`NONE` says "deliberately excluded"; `notes` says why. Both are needed:
the enum without the note re-raises "why is this one NONE?" every time
someone reads the table, and the note without the enum has nowhere to hang.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


@pytest.fixture
async def session():
    import fishsense_api.database  # noqa: F401  # pylint: disable=unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as open_session:
        yield open_session
    await engine.dispose()


def _camera_with_intrinsics(session, camera_id: int = 1):
    """`post_dive` validates camera + intrinsics on the effective row."""
    from fishsense_api.models.camera import Camera
    from fishsense_api.models.camera_intrinsics import CameraIntrinsics

    session.add(
        Camera(
            id=camera_id,
            name=f"FSL{camera_id:02d}",
            serial_number=f"SN{camera_id:06d}",
        )
    )
    session.add(
        CameraIntrinsics(
            id=camera_id,
            camera_id=camera_id,
            fx=1.0,
            fy=1.0,
            cx=1.0,
            cy=1.0,
            k1=0.0,
            k2=0.0,
            p1=0.0,
            p2=0.0,
            k3=0.0,
        )
    )


def _body(**overrides):
    from fishsense_api.models.dive import Dive

    fields = {
        "path": "/dev/null/parked",
        "dive_datetime": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "camera_id": 1,
    }
    fields.update(overrides)
    return Dive(**fields)


# --- the enum itself -------------------------------------------------------


def test_priority_has_none_member():
    from fishsense_api.models.priority import Priority

    assert Priority.NONE.value == "NONE"


def test_sdk_priority_mirrors_none():
    """The SDK enum is hand-mirrored; drift here is a wire-format break."""
    from fishsense_api.models.priority import Priority as ApiPriority
    from fishsense_api_sdk.models.priority import Priority as SdkPriority

    assert SdkPriority.NONE.value == "NONE"
    assert {p.value for p in SdkPriority} == {p.value for p in ApiPriority}


# --- persistence -----------------------------------------------------------


async def test_none_priority_round_trips(session):
    from fishsense_api.controllers.dive_controller import post_dive
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.priority import Priority

    _camera_with_intrinsics(session)
    await session.flush()

    dive_id = await post_dive(_body(priority=Priority.NONE), session=session)

    assert (await session.get(Dive, dive_id)).priority is Priority.NONE


async def test_notes_round_trip(session):
    from fishsense_api.controllers.dive_controller import post_dive
    from fishsense_api.models.dive import Dive

    _camera_with_intrinsics(session)
    await session.flush()

    note = "V-Slate 7; no scan exists, so this dive can never be calibrated."
    dive_id = await post_dive(_body(notes=note), session=session)

    assert (await session.get(Dive, dive_id)).notes == note


async def test_notes_survive_a_partial_repost(session):
    """The overlay in `post_dive` must not null `notes`.

    Same failure shape the overlay already guards for `dive_slate_id` and
    `calibration_dive_id`: a body that mentions only `priority` would
    otherwise wipe the reason the dive was parked in the first place.
    """
    from fishsense_api.controllers.dive_controller import post_dive
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.priority import Priority

    _camera_with_intrinsics(session)
    await session.flush()

    note = "V-Slate 7; no scan exists."
    dive_id = await post_dive(
        _body(priority=Priority.NONE, notes=note), session=session
    )

    # A later re-post that only speaks about priority.
    await post_dive(
        _body(priority=Priority.LOW),
        session=session,
    )

    row = await session.get(Dive, dive_id)
    assert row.priority is Priority.LOW
    assert row.notes == note


async def test_unknown_priority_still_422s(session):
    """The coercion guard must keep rejecting garbage once NONE exists."""
    from fastapi import HTTPException

    from fishsense_api.controllers.dive_controller import post_dive
    from fishsense_api.models.dive import Dive

    _camera_with_intrinsics(session)
    await session.flush()

    body = Dive.model_construct(
        path="/dev/null/bad",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        camera_id=1,
        priority="URGENT",
    )
    with pytest.raises(HTTPException) as exc:
        await post_dive(body, session=session)
    assert exc.value.status_code == 422


# --- the point of the enum: cohort exclusion -------------------------------


async def test_none_priority_dive_is_not_selected_by_a_cohort(session):
    """A parked dive must be invisible to the hourly selectors.

    Stage 0.1 is the broadest cohort (an image with no laser label at all),
    so a dive that escapes this one escapes nothing else either.
    """
    from tests_support.db import dive, image

    from fishsense_api.controllers.dive_cohort_controller import (
        select_next_for_laser_preprocessing,
    )
    from fishsense_api.models.priority import Priority

    session.add(dive(1, priority=Priority.NONE))
    session.add(image(10, 1))
    await session.flush()

    assert await select_next_for_laser_preprocessing(session=session) is None


# --- the notes endpoint ----------------------------------------------------
#
# A dedicated endpoint rather than a `post_dive` partial re-post: the species
# sync writes this, and a read-modify-write through the upsert would race any
# concurrent write to the same row.


async def test_set_notes_writes_the_note(session):
    from fishsense_api.controllers.dive_controller import set_notes
    from fishsense_api.models.dive import Dive
    from tests_support.db import dive

    session.add(dive(1))
    await session.flush()

    note = "Slate not in list (V-Slate 7, lost); cannot be calibrated."
    assert await set_notes(1, note, session=session) == 1
    assert (await session.get(Dive, 1)).notes == note


async def test_set_notes_can_clear(session):
    from fishsense_api.controllers.dive_controller import set_notes
    from fishsense_api.models.dive import Dive
    from tests_support.db import dive

    session.add(dive(1))
    await session.flush()

    await set_notes(1, "something", session=session)
    await set_notes(1, None, session=session)

    assert (await session.get(Dive, 1)).notes is None


async def test_set_notes_404s_on_a_missing_dive(session):
    from fastapi import HTTPException

    from fishsense_api.controllers.dive_controller import set_notes

    with pytest.raises(HTTPException) as exc:
        await set_notes(999, "x", session=session)
    assert exc.value.status_code == 404


async def test_set_notes_leaves_priority_and_slate_alone(session):
    """Writing a note must not be a back door into pipeline state.

    The species sync calls this automatically, so it must touch exactly one
    column — parking a dive stays an operator decision.
    """
    from fishsense_api.controllers.dive_controller import set_notes
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.priority import Priority
    from tests_support.db import dive

    session.add(dive(1, priority=Priority.HIGH, dive_slate_id=9))
    await session.flush()

    await set_notes(1, "unidentifiable slate", session=session)

    row = await session.get(Dive, 1)
    assert row.priority is Priority.HIGH
    assert row.dive_slate_id == 9

"""Tests for `CalibrationTarget` and the dive link that names one.

A `CalibrationTarget` is a planar target of known geometry that laser
extrinsics can be fitted against — the second producer described in
`docs/plans/checkerboard-laser-calibration.md`. A `DiveSlate` is the first
one; this table exists because a checkerboard is not a slate, and calling it
one would put the wrong word in the schema, in `dive_pipeline_status.slate_*`
and in Label Studio.

What these pin down:

  * `square_size_m` is required. It is the *only* number that sets the scale
    of every length the dive ultimately produces, and scale error is the one
    term reprojection residual cannot see — so a target without it must be
    impossible to store, not merely unusual.
  * `set_dive_calibration_target` / `clear_dive_calibration_target` manage
    `Dive.calibration_target_id`, with the missing-row guards the sibling
    slate and calibration-source endpoints have.

FK-less in-memory sqlite, same as `test_calibration_source_endpoints.py` — we
exercise the controller functions directly, not referential integrity.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

# Shared with the other controller tests — see `tests_support.db`.
from tests_support.db import (  # noqa: F401
    dive as _dive,
)

# The E4E board: 15 x 11 squares, so 14 x 10 interior corners. Only the
# interior corners are detectable, and they are what `solvePnP` is given.
E4E_ROWS = 10
E4E_COLS = 14


def _target(target_id: int = 1, **overrides):
    from fishsense_api.models.calibration_target import CalibrationTarget

    kwargs = {
        "id": target_id,
        "name": "E4E Checkerboard",
        "rows": E4E_ROWS,
        "cols": E4E_COLS,
        "square_size_m": 0.0254,
    }
    kwargs.update(overrides)
    return CalibrationTarget(**kwargs)


# ---------- the model ----------


@pytest.mark.parametrize("column", ["name", "rows", "cols", "square_size_m"])
def test_geometry_columns_are_not_nullable(column):
    """The database refuses a target that cannot actually be used.

    Asserted against the schema rather than the constructor because SQLModel
    skips pydantic validation on `table=True` models — so `CalibrationTarget`
    happily *constructs* without a square size, and NOT NULL is the only thing
    standing between that and a stored row.

    `square_size_m` is the one that matters. It alone sets the scale of every
    length the dive ultimately produces, and scale error is the term the
    reprojection residual provably cannot see, so a scale-less target would
    read as usable and silently not be.
    """
    from fishsense_api.models.calibration_target import CalibrationTarget

    assert CalibrationTarget.__table__.columns[column].nullable is False


async def test_a_target_without_a_square_size_cannot_be_stored(session):
    """The end the schema assertion above is a means to."""
    from sqlalchemy.exc import IntegrityError

    from fishsense_api.models.calibration_target import CalibrationTarget

    session.add(CalibrationTarget(id=1, name="Scale-less board", rows=10, cols=14))
    with pytest.raises(IntegrityError):
        await session.flush()


# ---------- the collection endpoints ----------


async def test_put_then_get_round_trips(session):
    from fishsense_api.controllers.calibration_target_controller import (
        get_calibration_targets,
        put_calibration_target,
    )

    await put_calibration_target(1, _target(), session=session)

    targets = await get_calibration_targets(session=session)
    assert [(t.name, t.rows, t.cols, t.square_size_m) for t in targets] == [
        ("E4E Checkerboard", E4E_ROWS, E4E_COLS, 0.0254)
    ]


async def test_put_upserts_on_the_id(session):
    """A re-measured board corrects its row rather than creating a second.

    Two rows for one physical board is the failure that matters here: dives
    would split across them and half the corpus would carry the superseded
    scale with nothing saying which was which.
    """
    from fishsense_api.controllers.calibration_target_controller import (
        get_calibration_targets,
        put_calibration_target,
    )

    await put_calibration_target(1, _target(), session=session)
    await put_calibration_target(1, _target(square_size_m=0.0249), session=session)

    targets = await get_calibration_targets(session=session)
    assert len(targets) == 1
    assert targets[0].square_size_m == 0.0249


# ---------- the dive link ----------


async def test_set_dive_calibration_target_links_the_dive(session):
    from fishsense_api.controllers.calibration_target_controller import (
        put_calibration_target,
    )
    from fishsense_api.controllers.dive_controller import set_dive_calibration_target

    session.add(_dive(1))
    await session.flush()
    await put_calibration_target(7, _target(7), session=session)

    assert await set_dive_calibration_target(1, 7, session=session) == 1

    from fishsense_api.models.dive import Dive

    assert (await session.get(Dive, 1)).calibration_target_id == 7


async def test_set_dive_calibration_target_404s_on_a_missing_dive(session):
    from fishsense_api.controllers.calibration_target_controller import (
        put_calibration_target,
    )
    from fishsense_api.controllers.dive_controller import set_dive_calibration_target

    await put_calibration_target(7, _target(7), session=session)

    with pytest.raises(HTTPException) as excinfo:
        await set_dive_calibration_target(999, 7, session=session)
    assert excinfo.value.status_code == 404


async def test_set_dive_calibration_target_404s_on_a_missing_target(session):
    """Refuse rather than store a dangling id.

    sqlite ignores the FK, so nothing else would catch this — and a dive
    pointing at a target that does not exist would sit in the calibration
    cohort and fail its resolver on every hourly firing.
    """
    from fishsense_api.controllers.dive_controller import set_dive_calibration_target

    session.add(_dive(1))
    await session.flush()

    with pytest.raises(HTTPException) as excinfo:
        await set_dive_calibration_target(1, 404, session=session)
    assert excinfo.value.status_code == 404


async def test_clear_dive_calibration_target_is_idempotent(session):
    from fishsense_api.controllers.calibration_target_controller import (
        put_calibration_target,
    )
    from fishsense_api.controllers.dive_controller import (
        clear_dive_calibration_target,
        set_dive_calibration_target,
    )
    from fishsense_api.models.dive import Dive

    session.add(_dive(1))
    await session.flush()
    await put_calibration_target(7, _target(7), session=session)
    await set_dive_calibration_target(1, 7, session=session)

    await clear_dive_calibration_target(1, session=session)
    assert (await session.get(Dive, 1)).calibration_target_id is None

    # Clearing an already-null link is a no-op, not an error.
    await clear_dive_calibration_target(1, session=session)
    assert (await session.get(Dive, 1)).calibration_target_id is None


async def test_clear_dive_calibration_target_404s_on_a_missing_dive(session):
    from fishsense_api.controllers.dive_controller import clear_dive_calibration_target

    with pytest.raises(HTTPException) as excinfo:
        await clear_dive_calibration_target(999, session=session)
    assert excinfo.value.status_code == 404


async def test_the_slate_link_and_the_target_link_are_independent(session):
    """A dive can carry both without either overwriting the other.

    They answer different questions — "which slate template" and "which
    planar target" — and rig 04 of the 2023.08.18 set is the case that makes
    that concrete: it shot a real dive slate, so it calibrates through the
    existing path, while its siblings shot a checkerboard.
    """
    from fishsense_api.controllers.calibration_target_controller import (
        put_calibration_target,
    )
    from fishsense_api.controllers.dive_controller import (
        set_dive_calibration_target,
        set_dive_slate,
    )
    from fishsense_api.models.dive import Dive
    from fishsense_api.models.dive_slate import DiveSlate

    session.add(_dive(1))
    session.add(DiveSlate(id=3, name="H-Slate", path="/dev/null/h", dpi=300))
    await session.flush()
    await put_calibration_target(7, _target(7), session=session)

    await set_dive_slate(1, 3, session=session)
    await set_dive_calibration_target(1, 7, session=session)

    stored = await session.get(Dive, 1)
    assert (stored.dive_slate_id, stored.calibration_target_id) == (3, 7)

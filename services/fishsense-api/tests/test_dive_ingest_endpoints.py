"""Tests for `POST /api/v1/dives/` — the dive-creation half of ingest.

Before this endpoint there was no way to create a `Dive` through this repo at
all; rows came from `UCSD-E4E/fishsense-data-processing-spider` and were copied
in wholesale by commit `9e5bc64`. See `docs/plans/dive-image-ingestion.md` §0
for the conventions that archaeology established.

Two behaviours carry the weight here:

  * **Natural-key upsert on `path`.** `Dive.path` is unique, and a blind
    `session.merge` with `id=None` always INSERTs — which is exactly the #347
    duplicate-key 500. Ingest re-runs are the normal case (resume after a
    partial scan, and the finalize step re-POSTs the same path to flip
    `priority`), so upsert-by-path is load-bearing, not a nicety.
  * **Validation that fails loudly.** A dive whose camera has no
    `CameraIntrinsics` cannot ever be measured by stage 14, and nothing
    downstream errors — the dive just silently never reaches `measured`. The
    endpoint refuses instead.

FK-less in-memory sqlite, controller functions called directly — same shape as
test_calibration_source_endpoints.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, select
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


async def _seed_camera(session, camera_id: int = 1, *, with_intrinsics: bool = True):
    from fishsense_api.models.camera import Camera  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.camera_intrinsics import (  # pylint: disable=import-outside-toplevel
        CameraIntrinsics,
    )

    session.add(
        Camera(id=camera_id, serial_number=f"SN{camera_id}", name=f"FSL-0{camera_id}")
    )
    await session.flush()
    if with_intrinsics:
        session.add(
            CameraIntrinsics(
                camera_id=camera_id,
                camera_matrix=[[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]],
                distortion_coefficients=[-0.05, 0.01, 0.0, 0.0, 0.0],
            )
        )
        await session.flush()


def _dive(path: str, **kwargs):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    kwargs.setdefault("camera_id", 1)
    kwargs.setdefault("priority", Priority.LOW)
    return Dive(
        path=path,
        dive_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
        **kwargs,
    )


# ── creation ──────────────────────────────────────────────────────────


async def test_post_dive_creates_a_row_and_returns_its_id(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    await _seed_camera(session)

    dive_id = await post_dive(_dive("2024 REEF/082124_Alligator_FSL06"), session=session)

    assert isinstance(dive_id, int)
    row = (await session.exec(select(Dive).where(Dive.id == dive_id))).first()
    assert row is not None
    assert row.path == "2024 REEF/082124_Alligator_FSL06"


async def test_post_dive_is_an_upsert_on_path_not_a_duplicate_insert(session):
    """The #347 regression, in its dive-shaped form.

    Ingest re-runs the same path routinely: resuming a partial scan, and the
    finalize step re-POSTing to flip `priority` LOW -> HIGH. A blind
    `session.merge(id=None)` INSERTs and trips the unique index on `path`.
    """
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    await _seed_camera(session)
    path = "2024 REEF/082124_Alligator_FSL06"

    first = await post_dive(_dive(path), session=session)
    second = await post_dive(
        _dive(path, priority=Priority.HIGH, name="082124_Alligator_FSL06"),
        session=session,
    )

    assert first == second
    rows = (await session.exec(select(Dive).where(Dive.path == path))).all()
    assert len(rows) == 1
    # The second POST is the finalize step: its values must win.
    assert rows[0].priority == Priority.HIGH
    assert rows[0].name == "082124_Alligator_FSL06"


# ── validation ────────────────────────────────────────────────────────


async def test_post_dive_rejects_a_missing_camera_id(session):
    """No camera means no intrinsics means stage 14 can never measure it."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )

    with pytest.raises(HTTPException) as exc:
        await post_dive(_dive("d/1", camera_id=None), session=session)
    assert exc.value.status_code == 422


async def test_post_dive_rejects_an_unknown_camera_id(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )

    with pytest.raises(HTTPException) as exc:
        await post_dive(_dive("d/1", camera_id=999), session=session)
    assert exc.value.status_code == 422


async def test_post_dive_rejects_a_camera_without_intrinsics(session):
    """The silent-failure case this endpoint exists to make loud: the dive would
    be created happily and then never reach `measured`, with no error anywhere."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )

    await _seed_camera(session, with_intrinsics=False)

    with pytest.raises(HTTPException) as exc:
        await post_dive(_dive("d/1"), session=session)
    assert exc.value.status_code == 422
    assert "intrinsics" in exc.value.detail.lower()


def test_an_over_long_dive_path_is_rejected_when_the_body_is_validated():
    """Where the 255 guard actually bites.

    **SQLModel `table=True` models do not validate on `__init__`** — plain
    construction happily accepts a 256-char path. Only `model_validate` (which
    is what FastAPI runs on a request body) enforces `max_length`. So an HTTP
    caller gets a 422 from request validation, but any in-process caller that
    constructs a `Dive` directly sails straight past it.

    That asymmetry is exactly why the endpoint keeps its own explicit check
    (next test) rather than trusting the model.
    """
    from pydantic import ValidationError  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    _dive("x" * 256)  # constructing is NOT validated — no raise

    with pytest.raises(ValidationError):
        Dive.model_validate(
            {
                "path": "x" * 256,
                "dive_datetime": datetime(2024, 8, 21, tzinfo=timezone.utc),
                "camera_id": 1,
            }
        )


async def test_post_dive_rejects_an_over_long_path_that_bypassed_validation(session):
    """`model_construct` skips pydantic. Postgres would reject the value, but
    sqlite silently stores it, and a truncated `Dive.path` no longer resolves
    on the NAS — so the endpoint checks explicitly, before re-validating."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    await _seed_camera(session)
    smuggled = Dive.model_construct(
        path="x" * 256,
        dive_datetime=datetime(2024, 8, 21, tzinfo=timezone.utc),
        camera_id=1,
    )

    with pytest.raises(HTTPException) as exc:
        await post_dive(smuggled, session=session)
    assert exc.value.status_code == 422
    assert "255" in exc.value.detail


async def test_post_dive_rejects_a_self_referential_calibration_source(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )

    await _seed_camera(session)
    dive_id = await post_dive(_dive("d/1"), session=session)

    with pytest.raises(HTTPException) as exc:
        await post_dive(_dive("d/1", calibration_dive_id=dive_id), session=session)
    assert exc.value.status_code == 422


async def test_post_dive_rejects_an_unknown_calibration_source(session):
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )

    await _seed_camera(session)

    with pytest.raises(HTTPException) as exc:
        await post_dive(_dive("d/1", calibration_dive_id=4242), session=session)
    assert exc.value.status_code == 422


async def test_post_dive_accepts_a_valid_calibration_source(session):
    """A fish-only dive borrowing a sibling slate dive's calibration is the
    whole point of `calibration_dive_id` — it must not be blocked."""
    from fishsense_api.controllers.dive_controller import (  # pylint: disable=import-outside-toplevel
        post_dive,
    )
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel

    await _seed_camera(session)
    slate_dive = await post_dive(_dive("d/slate"), session=session)

    fish_dive = await post_dive(
        _dive("d/fish", calibration_dive_id=slate_dive), session=session
    )

    row = (await session.exec(select(Dive).where(Dive.id == fish_dive))).first()
    assert row.calibration_dive_id == slate_dive

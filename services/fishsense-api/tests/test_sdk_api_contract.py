"""Contract tests: the SDK's ingest calls against the real FastAPI app.

The SDK's own client tests mock `_post`, so they pin the URL and the payload
shape the client *intends* to send — but nothing checks the API would accept
it. A renamed route, a field the endpoint rejects, or a response shape the
client can't parse would leave both suites green and fail only in production.

`test_sdk_drift.py` covers the other half of the same seam (field-name and type
parity between the SQLModel and the SDK's pydantic mirror). This covers the
wire: real routing, real request validation, real serialization, driven by the
actual client methods over `httpx.ASGITransport` — no server, no network.

The app is not entered as a context manager; that would run the real lifespan
(`create_all` against Postgres, then `run_alembic_upgrade`).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import httpx
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

CK_A = "45dc5a454b35601b9dafabf24822195d"
CK_B = "0b7cd4da72d54172f1f9daf40ce4047f"


@pytest.fixture
async def sdk():
    """A real `DiveClient`/`ImageClient` pair wired to the in-process app."""
    os.environ.setdefault("E4EFS_POSTGRES__HOST", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PORT", "5432")
    os.environ.setdefault("E4EFS_POSTGRES__USERNAME", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PASSWORD", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__DATABASE", "ignored")

    import fishsense_api.controllers  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
    from fishsense_api.database import (  # pylint: disable=import-outside-toplevel
        get_async_session,
    )
    from fishsense_api.models.camera import Camera  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.camera_intrinsics import (  # pylint: disable=import-outside-toplevel
        CameraIntrinsics,
    )
    from fishsense_api.server import app  # pylint: disable=import-outside-toplevel
    from fishsense_api_sdk.clients.dive_client import (  # pylint: disable=import-outside-toplevel
        DiveClient,
    )
    from fishsense_api_sdk.clients.image_client import (  # pylint: disable=import-outside-toplevel
        ImageClient,
    )

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = factory()
    session.add(Camera(id=1, serial_number="BJ6C67989", name="FSL-07"))
    await session.flush()
    session.add(
        CameraIntrinsics(
            camera_id=1,
            camera_matrix=[[3000.0, 0.0, 2000.0], [0.0, 3000.0, 1500.0], [0.0, 0.0, 1.0]],
            distortion_coefficients=[-0.05, 0.01, 0.0, 0.0, 0.0],
        )
    )
    await session.flush()

    async def _override():
        yield session

    app.dependency_overrides[get_async_session] = _override

    transport = httpx.ASGITransport(app=app)
    kwargs = {
        "base_url": "http://api.test",
        "username": None,
        "password": None,
        "timeout": 10,
        "semaphore": asyncio.Semaphore(4),
        "transport": transport,
    }
    async with DiveClient(**kwargs) as dives, ImageClient(**kwargs) as images:
        yield dives, images

    app.dependency_overrides.pop(get_async_session, None)
    await session.close()
    await engine.dispose()


def _sdk_dive(path: str, **kwargs):
    from fishsense_api_sdk.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api_sdk.models.priority import (  # pylint: disable=import-outside-toplevel
        Priority,
    )

    body = {
        "id": None,
        "name": None,
        "path": path,
        "dive_datetime": datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
        "priority": Priority.LOW,
        "flip_dive_slate": False,
        "camera_id": 1,
        "dive_slate_id": None,
        "calibration_dive_id": None,
    }
    body.update(kwargs)
    return Dive(**body)


def _sdk_image(path: str, checksum: str, **kwargs):
    from fishsense_api_sdk.models.image import Image  # pylint: disable=import-outside-toplevel

    body = {
        "id": None,
        "path": path,
        "taken_datetime": datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
        "checksum": checksum,
        "is_canonical": False,
        "dive_id": None,
        "camera_id": 1,
    }
    body.update(kwargs)
    return Image(**body)


async def test_dives_post_is_accepted_by_the_api(sdk):
    """The SDK serializes `dive_datetime` and `priority` itself; the endpoint
    has to accept exactly that encoding."""
    dives, _ = sdk

    dive_id = await dives.post(_sdk_dive("2024 REEF/082124_FSL06"))

    assert isinstance(dive_id, int)
    fetched = await dives.get(dive_id=dive_id)
    assert fetched.path == "2024 REEF/082124_FSL06"


async def test_images_post_is_accepted_and_reaches_the_right_dive(sdk):
    dives, images = sdk
    dive_id = await dives.post(_sdk_dive("d/7"))

    image_id = await images.post(dive_id, _sdk_image("d/7/P8210001.ORF", CK_A))

    assert isinstance(image_id, int)
    fetched = await images.get(image_id=image_id)
    assert fetched.dive_id == dive_id
    assert fetched.checksum == CK_A


async def test_the_sdk_does_not_override_server_computed_canonicality(sdk):
    """`Image.is_canonical` is required on the SDK model, so every instance
    carries a value. If `post` shipped it, the second copy of a frame would
    claim canonical too — which the DB now rejects outright. End to end, the
    server must be the one deciding."""
    dives, images = sdk
    first_dive = await dives.post(_sdk_dive("d/64"))
    second_dive = await dives.post(_sdk_dive("d/66"))

    first = await images.post(first_dive, _sdk_image("d/64/P1.ORF", CK_A))
    dupe = await images.post(second_dive, _sdk_image("d/66/P1.ORF", CK_A))

    assert (await images.get(image_id=first)).is_canonical is True
    assert (await images.get(image_id=dupe)).is_canonical is False


async def test_lookup_checksums_round_trips_through_the_real_endpoint(sdk):
    """The response is a nested dict the client parses raw — a shape change on
    either side breaks silently under mocks."""
    dives, images = sdk
    dive_id = await dives.post(_sdk_dive("d/64"))
    await images.post(dive_id, _sdk_image("d/64/P1.ORF", CK_A))

    result = await images.lookup_checksums([CK_A, CK_B])

    assert result[CK_B] == []
    assert result[CK_A][0]["dive_id"] == dive_id
    assert result[CK_A][0]["is_canonical"] is True


async def test_reposting_a_dive_through_the_sdk_preserves_unmentioned_fields(sdk):
    """The SDK sends `exclude_unset=True`, so a partially-populated `Dive`
    reaches the API as a genuinely partial body. This is the combination that
    triggers the destructive-upsert bug in production, and neither suite could
    see it alone."""
    from fishsense_api_sdk.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api_sdk.models.priority import (  # pylint: disable=import-outside-toplevel
        Priority,
    )

    dives, _ = sdk
    dive_id = await dives.post(
        _sdk_dive("d/fish", name="fish dive", dive_slate_id=3)
    )

    await dives.post(
        Dive.model_construct(
            path="d/fish",
            dive_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
            camera_id=1,
            priority=Priority.HIGH,
        )
    )

    row = await dives.get(dive_id=dive_id)
    assert row.priority == Priority.HIGH
    assert row.name == "fish dive"
    assert row.dive_slate_id == 3

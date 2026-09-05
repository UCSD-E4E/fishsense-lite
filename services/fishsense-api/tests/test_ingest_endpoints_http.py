"""App-level tests for the ingest write endpoints.

Every other ingest test calls the controller function directly, which is fast
and precise but skips three things that only exist at the app boundary:

  * **Route registration.** A controller that isn't imported in
    `controllers/__init__.py` silently registers nothing, and a direct-call
    test would still pass.
  * **Request-body validation.** `Dive.path` is `max_length=255`, but SQLModel
    `table=True` models don't validate on `__init__` — only `model_validate`
    does, which is what FastAPI runs on a request body. So the 255 guard only
    actually fires here.
  * **Response serialization.** `post_checksum_lookup` returns a nested
    `Dict[str, List[Dict[str, Any]]]`. FastAPI builds a response model from
    that annotation, and a shape it can't encode would 500 at runtime while
    every direct-call test stayed green.

The session dependency is overridden onto in-memory sqlite. The app is used
*without* `TestClient`'s context manager on purpose: entering it would run the
real lifespan, which calls `create_all` against Postgres and then
`run_alembic_upgrade`.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from tests_support.app import (
    seed_camera_with_intrinsics,
    seed_placeholder_settings,
)

CK_A = "45dc5a454b35601b9dafabf24822195d"
CK_B = "0b7cd4da72d54172f1f9daf40ce4047f"

_DIVE_DT = "2024-08-21T08:56:51Z"


@pytest.fixture
async def client():
    from fastapi.testclient import TestClient
    seed_placeholder_settings()

    import fishsense_api.controllers  # noqa: F401  pylint: disable=unused-import
    from fishsense_api.database import (
        get_async_session,
    )
    from fishsense_api.server import app

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    session = factory()
    await seed_camera_with_intrinsics(session)

    async def _override():
        yield session

    app.dependency_overrides[get_async_session] = _override
    # No `with` — see the module docstring.
    yield TestClient(app)
    app.dependency_overrides.pop(get_async_session, None)

    await session.close()
    await engine.dispose()


def _dive_body(path: str, **kwargs) -> dict:
    body = {"path": path, "dive_datetime": _DIVE_DT, "camera_id": 1}
    body.update(kwargs)
    return body


def _image_body(path: str, checksum: str, **kwargs) -> dict:
    body = {"path": path, "checksum": checksum, "taken_datetime": _DIVE_DT}
    body.update(kwargs)
    return body


# ── the routes exist and are reachable ────────────────────────────────


def test_the_three_ingest_routes_are_registered(client):
    """`controllers/__init__.py` is the route registry — a controller missing
    from it registers nothing, and every direct-call test would still pass."""
    paths = {
        (route.path, method)
        for route in client.app.routes
        for method in getattr(route, "methods", set()) or set()
    }
    assert ("/api/v1/dives/", "POST") in paths
    assert ("/api/v1/dives/{dive_id}/images/", "POST") in paths
    assert ("/api/v1/images/checksums/lookup", "POST") in paths


def test_the_checksum_lookup_route_does_not_shadow_the_single_checksum_get(client):
    """`/images/checksums/lookup` sits next to `/images/checksum/{checksum}`
    and `/images/{image_id}`. A path that got swallowed by `{image_id}` would
    try to coerce "checksums" to an int and 422."""
    response = client.post("/api/v1/images/checksums/lookup", json=[CK_A])
    assert response.status_code == 200


# ── request validation at the boundary ────────────────────────────────


def test_an_over_long_dive_path_is_a_422_over_http(client):
    """Where `max_length=255` actually bites: FastAPI runs `model_validate` on
    the request body. A direct call constructing a `Dive` sails past it, which
    is why the endpoint keeps its own check too."""
    response = client.post("/api/v1/dives/", json=_dive_body("x" * 256))
    assert response.status_code == 422


def test_a_dive_body_missing_required_fields_is_a_422_over_http(client):
    response = client.post("/api/v1/dives/", json={"path": "d/1"})
    assert response.status_code == 422


# ── round trip ────────────────────────────────────────────────────────


def test_dive_then_images_then_lookup_round_trips_over_http(client):
    """One pass of what ingest actually does, through the real stack."""
    dive = client.post("/api/v1/dives/", json=_dive_body("2024 REEF/082124_FSL06"))
    assert dive.status_code == 201, dive.text
    dive_id = dive.json()

    first = client.post(
        f"/api/v1/dives/{dive_id}/images/", json=_image_body("d/P1.ORF", CK_A)
    )
    assert first.status_code == 201, first.text

    # A second dive holding the same frame — the dives-64/66 shape.
    other = client.post("/api/v1/dives/", json=_dive_body("2024 REEF/082124_FSL06_copy"))
    dupe = client.post(
        f"/api/v1/dives/{other.json()}/images/", json=_image_body("d2/P1.ORF", CK_A)
    )
    assert dupe.status_code == 201

    lookup = client.post(
        "/api/v1/images/checksums/lookup", json=[CK_A, CK_B]
    )
    assert lookup.status_code == 200, lookup.text
    body = lookup.json()

    # Serialization of the nested Dict[str, List[Dict[str, Any]]] survives.
    assert body[CK_B] == []
    assert {hit["dive_id"] for hit in body[CK_A]} == {dive_id, other.json()}
    assert sorted(hit["is_canonical"] for hit in body[CK_A]) == [False, True]


def test_reposting_a_dive_over_http_preserves_unmentioned_fields(client):
    """The destructive-upsert regression, through the real request path — the
    body genuinely omits the keys rather than relying on `model_fields_set`
    being set up correctly by a direct call."""
    created = client.post(
        "/api/v1/dives/",
        json=_dive_body("d/fish", name="fish dive", dive_slate_id=3),
    )
    dive_id = created.json()

    client.post("/api/v1/dives/", json=_dive_body("d/fish", priority="HIGH"))

    row = client.get(f"/api/v1/dives/{dive_id}").json()
    assert row["priority"] == "HIGH"
    assert row["name"] == "fish dive"
    assert row["dive_slate_id"] == 3


def test_priority_is_stored_as_the_enum_not_the_raw_json_string(client):
    """SQLModel `table=True` skips pydantic coercion, so FastAPI hands the
    endpoint a `Dive` whose `priority` is still the raw JSON `str`. Coercing it
    up front is what keeps `jsonable_encoder` from emitting a pydantic
    serializer warning on every request — run the suite with `-W error` and an
    uncoerced body fails here."""
    from fishsense_api.models.priority import Priority

    created = client.post("/api/v1/dives/", json=_dive_body("d/p", priority="HIGH"))
    assert created.status_code == 201

    listed = client.get("/api/v1/dives/").json()
    assert listed[0]["priority"] == Priority.HIGH.value


def test_an_unknown_priority_is_a_422_not_a_500(client):
    response = client.post("/api/v1/dives/", json=_dive_body("d/p", priority="URGENT"))
    assert response.status_code == 422

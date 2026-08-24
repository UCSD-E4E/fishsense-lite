"""Helpers for the tests that stand up the FastAPI app itself.

Three things recur across those tests and had drifted into per-file copies:
seeding dynaconf placeholders, seeding the camera an ingest request needs, and
resolving a path against the route table.

The env seeding in particular is not boilerplate that could be deleted — it is
the documented dynaconf gotcha. Dynaconf validates EVERY validator on first
attribute access of `settings`, so importing any module that touches config
requires every unrelated setting to be present, whether or not the test uses
it. `setdefault`, never assignment: the Postgres integration tests point the
same variables at a real scratch database, and clobbering them there would
silently aim an in-memory suite at it.
"""

from __future__ import annotations

import os

__all__ = ["seed_placeholder_settings", "seed_camera_with_intrinsics", "resolve_route"]

_PLACEHOLDER_SETTINGS = {
    "E4EFS_POSTGRES__HOST": "ignored",
    "E4EFS_POSTGRES__PORT": "5432",
    "E4EFS_POSTGRES__USERNAME": "ignored",
    "E4EFS_POSTGRES__PASSWORD": "ignored",
    "E4EFS_POSTGRES__DATABASE": "ignored",
}


def seed_placeholder_settings() -> None:
    """Give dynaconf something to validate before any controller is imported."""
    for key, value in _PLACEHOLDER_SETTINGS.items():
        os.environ.setdefault(key, value)


async def seed_camera_with_intrinsics(session) -> None:
    """The camera an ingest/measurement request resolves against.

    Real-ish values: an Olympus-scale 3000 px focal length on a 4000x3000
    frame, which keeps any projection a test performs in a plausible range.
    """
    from fishsense_api.models.camera import Camera
    from fishsense_api.models.camera_intrinsics import CameraIntrinsics

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


def resolve_route(app, path: str) -> str | None:
    """The name of the endpoint function `path` matches, in declaration order.

    Returns the endpoint's `__name__`, not the route template: the assertion
    these tests want to make is "this URL reaches THAT handler", and the
    template alone would not distinguish two handlers registered on the same
    path.

    The question the disambiguation tests exist to ask: `/dives/select-next/...`
    must win over `/dives/{dive_id}`, and across modules that ordering is
    decided by import order in `controllers/__init__.py`. Get it wrong and
    every cohort poll 422s in prod while the unit tests stay green.
    """
    from starlette.routing import Match

    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "path_params": {},
        "route_path": path,
        "headers": [],
    }
    for route in app.routes:
        if not hasattr(route, "matches"):
            continue
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return route.endpoint.__name__
    return None

"""Regression guard for dive-controller route ordering.

The `select-next/...` family must be declared before
`/api/v1/dives/{dive_id}` in `dive_controller.py` because FastAPI
matches in declaration order: a `/dives/select-next/laser-preprocessing/`
request would otherwise try to coerce "select-next" into the
`{dive_id}: int` path param and 422.

This test enumerates every cohort selector route, including the new
`dive-frame-clustering` and renamed `species-preprocessing` endpoints.
A reorder, a typo in the URL, or a missing route registration fails
the matching parametrize case.
"""

from __future__ import annotations

import os

import pytest
from starlette.routing import Match


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("E4EFS_POSTGRES__HOST", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PORT", "5432")
    os.environ.setdefault("E4EFS_POSTGRES__USERNAME", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__PASSWORD", "ignored")
    os.environ.setdefault("E4EFS_POSTGRES__DATABASE", "ignored")

    import fishsense_api.controllers.dive_controller  # noqa: F401, pylint: disable=import-outside-toplevel,unused-import
    from fishsense_api.server import app  # pylint: disable=import-outside-toplevel

    return app


def _resolve(app, path: str) -> str | None:
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


@pytest.mark.parametrize(
    "path,expected_endpoint",
    [
        # Cohort selectors — all must resolve to their own handlers,
        # NOT to `get_dive` (which would 422 on the non-int path
        # segment). Order in the controller is: select-next/...
        # routes first, then /{dive_id} catch-all. See dive_controller
        # docstring for the rationale.
        (
            "/api/v1/dives/select-next/laser-preprocessing/",
            "select_next_for_laser_preprocessing",
        ),
        (
            "/api/v1/dives/select-next/dive-frame-clustering/",
            "select_next_for_dive_frame_clustering",
        ),
        (
            "/api/v1/dives/select-next/species-preprocessing/",
            "select_next_for_species_preprocessing",
        ),
        (
            "/api/v1/dives/select-next/headtail-preprocessing/",
            "select_next_for_headtail_preprocessing",
        ),
        (
            "/api/v1/dives/select-next/slate-preprocessing/",
            "select_next_for_slate_preprocessing",
        ),
        (
            "/api/v1/dives/select-next/laser-calibration/",
            "select_next_for_laser_calibration",
        ),
        (
            "/api/v1/dives/select-next/measure-fish/",
            "select_next_for_measure_fish",
        ),
        # Numeric dive_id catch-all still resolves correctly even
        # though it's declared last.
        ("/api/v1/dives/123", "get_dive"),
    ],
)
def test_dive_route_resolves_to_expected_endpoint(app, path, expected_endpoint):
    assert _resolve(app, path) == expected_endpoint

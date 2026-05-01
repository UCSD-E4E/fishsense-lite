"""Regression guard for label-controller route ordering.

Starlette's default path converter compiles `{image_id}` to `[^/]+`, so
`/api/v1/labels/laser/label-studio-project-ids` would match
`/api/v1/labels/laser/{image_id}` and FastAPI would 422 on int
validation. The literal route is registered first in
`label_controller.py`; this test fails if anyone reorders.
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

    import fishsense_api.controllers.label_controller  # noqa: F401, pylint: disable=import-outside-toplevel,unused-import
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
        (
            "/api/v1/labels/laser/label-studio-project-ids",
            "get_laser_label_studio_project_ids",
        ),
        (
            "/api/v1/labels/headtail/label-studio-project-ids",
            "get_headtail_label_studio_project_ids",
        ),
        (
            "/api/v1/labels/species/label-studio-project-ids",
            "get_species_label_studio_project_ids",
        ),
        (
            "/api/v1/labels/dive-slate/label-studio-project-ids",
            "get_dive_slate_label_studio_project_ids",
        ),
        ("/api/v1/labels/laser/123", "get_laser_label"),
        ("/api/v1/labels/headtail/45", "get_headtail_label"),
        ("/api/v1/labels/species/67", "get_species_label"),
        ("/api/v1/labels/dive-slate/89", "get_dive_slate_label"),
        (
            "/api/v1/labels/laser/label-studio/777",
            "get_laser_label_by_label_studio_id",
        ),
        (
            "/api/v1/labels/headtail/label-studio/888",
            "get_headtail_label_by_label_studio_id",
        ),
    ],
)
def test_label_route_resolves_to_expected_endpoint(app, path, expected_endpoint):
    assert _resolve(app, path) == expected_endpoint

"""Regression guard for label-controller route ordering.

Starlette's default path converter compiles `{image_id}` to `[^/]+`, so
`/api/v1/labels/laser/label-studio-project-ids` would match
`/api/v1/labels/laser/{image_id}` and FastAPI would 422 on int
validation. The literal route is registered first in
`label_controller.py`; this test fails if anyone reorders.
"""

from __future__ import annotations

import pytest
from tests_support.app import resolve_route, seed_placeholder_settings


@pytest.fixture(scope="module")
def app():
    seed_placeholder_settings()

    import fishsense_api.controllers.label_controller  # noqa: F401, pylint: disable=unused-import
    from fishsense_api.server import app

    return app


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
        (
            "/api/v1/labels/laser/dives-with-complete-labeling",
            "get_dives_with_complete_laser_labeling",
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
    assert resolve_route(app, path) == expected_endpoint

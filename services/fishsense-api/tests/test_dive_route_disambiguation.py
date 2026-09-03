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

import pytest
from tests_support.app import resolve_route, seed_placeholder_settings


@pytest.fixture(scope="module")
def app():
    seed_placeholder_settings()

    import fishsense_api.controllers.dive_controller  # noqa: F401, pylint: disable=unused-import
    from fishsense_api.server import app

    return app


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
        # The three model-assisted (prediction) selectors. These live in
        # `dive_prediction_cohort_controller`, a *second* module that must also
        # be imported before `dive_controller`; until that split they were not
        # covered here at all, so an import reorder would have 422'd every
        # detector poll in prod with the suite still green.
        (
            "/api/v1/dives/select-next/laser-prediction/",
            "select_next_for_laser_prediction",
        ),
        (
            "/api/v1/dives/select-next/headtail-prediction/",
            "select_next_for_headtail_prediction",
        ),
        (
            "/api/v1/dives/select-next/slate-prediction/",
            "select_next_for_slate_prediction",
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
    assert resolve_route(app, path) == expected_endpoint

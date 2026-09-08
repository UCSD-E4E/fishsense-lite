"""Every label route still reaches the handler it is supposed to.

This is the safety net for splitting `label_controller`. Controllers register
their routes as a side effect of being imported from `controllers/__init__.py`,
so moving a handler to a new module has two ways to fail silently: the module
is never imported and the route vanishes, or it is imported in a position where
an earlier, looser route shadows it. FastAPI matches in declaration order, and
across modules that order is import order in the registry.

Neither failure is visible to a test that calls the handler function directly,
which is how most of the label suite is written.

The `needs-reprocess` verbs are listed explicitly because they are the ones
that moved, and because they share their path prefix with the per-dive label
GETs: `/dives/{id}/labels/laser` and `/dives/{id}/labels/laser/needs-reprocess`
must not be able to swallow one another.
"""

from __future__ import annotations

import pytest

from tests_support.app import resolve_route, seed_placeholder_settings


@pytest.fixture(scope="module")
def app():
    seed_placeholder_settings()
    import fishsense_api.controllers  # noqa: F401  pylint: disable=unused-import
    from fishsense_api.server import app as fastapi_app

    return fastapi_app


KINDS = ["laser", "species", "headtail", "dive-slate"]

# `dive-slate` breaks the pattern: its handlers are named `dive_slate_*` while
# its URL segment keeps the hyphen.
HANDLER_STEM = {
    "laser": "laser",
    "species": "species",
    "headtail": "headtail",
    "dive-slate": "dive_slate",
}


@pytest.mark.parametrize("kind", KINDS)
def test_needs_reprocess_verbs_reach_their_handlers(app, kind):
    stem = HANDLER_STEM[kind]
    path = f"/api/v1/dives/1/labels/{kind}/needs-reprocess"

    assert resolve_route(app, path, "PUT") == f"set_{stem}_labels_needs_reprocess"
    assert resolve_route(app, path, "DELETE") == f"clear_{stem}_labels_needs_reprocess"


@pytest.mark.parametrize("kind", KINDS)
def test_the_per_dive_label_get_is_not_shadowed(app, kind):
    """The sibling route the reprocess verbs sit underneath."""
    stem = HANDLER_STEM[kind]
    resolved = resolve_route(app, f"/api/v1/dives/1/labels/{kind}")
    assert resolved == f"get_{stem}_labels_for_dive"


@pytest.mark.parametrize("kind", KINDS)
def test_the_per_image_label_verbs_reach_their_handlers(app, kind):
    stem = HANDLER_STEM[kind]
    path = f"/api/v1/labels/{kind}/5"

    assert resolve_route(app, path) == f"get_{stem}_label"
    assert resolve_route(app, path, "PUT") == f"put_{stem}_label"


@pytest.mark.parametrize("kind", KINDS)
def test_the_project_ids_route_is_not_swallowed_by_the_image_id_route(app, kind):
    """`label-studio-project-ids` is a literal that has to beat `{image_id}`."""
    stem = HANDLER_STEM[kind]
    resolved = resolve_route(app, f"/api/v1/labels/{kind}/label-studio-project-ids")
    assert resolved == f"get_{stem}_label_studio_project_ids"

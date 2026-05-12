"""Shared fixtures for api-workflow-worker tests.

Dynaconf eagerly validates every `Validator` on first attribute access of
`settings`, not lazily per-setting (see CLAUDE.md). Any test that imports
an activity module — even one that never calls into `get_fs_client` or
`get_ls_client` — risks tripping this if the worker process happens to
read settings during import. The fixture seeds placeholder values for
every required setting so unrelated validators don't reject the test
process.

Integration-marked tests opt out of the placeholder override and run
against the real env vars set by `deploy/compose.local.yml` (real LS,
real fishsense-api). This is what `@pytest.mark.integration` is wired
to in the section below.
"""

import uuid

import pytest
from label_studio_sdk.client import LabelStudio

from fishsense_api_workflow_worker.activities.utils import get_ls_client


def _is_integration_test(request: pytest.FixtureRequest) -> bool:
    return request.node.get_closest_marker("integration") is not None


@pytest.fixture(autouse=True)
def configure_worker_settings(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
):
    # Dynaconf eagerly validates EVERY validator on first attribute
    # access of `settings`, so even tests that never read temporal.host
    # need it set or the test process fails at import time. Placeholders
    # for the unused-by-this-test backends below. Set temporal.port /
    # .tls explicitly too (not just .host): they have validator defaults,
    # but `settings.reload()` on the integration path doesn't re-apply
    # those, so a test that reaches Temporal (e.g. test_scale_down_query_integration)
    # would otherwise hit a missing-key error on `.port` / `.tls`.
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_TEMPORAL__PORT", "7233")
    monkeypatch.setenv("E4EFS_TEMPORAL__TLS", "false")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "http://nas.example.com")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "unused")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "unused")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.example.com")

    if _is_integration_test(request):
        # LS + static-files come from compose.local.yml's `dev` service
        # env (real LS endpoint, real bootstrap token). DON'T override
        # them — that's the point of integration tests. Force a settings
        # reload so any earlier placeholder-based attribute access is
        # invalidated.
        from fishsense_api_workflow_worker import config as cfg  # pylint: disable=import-outside-toplevel

        cfg.settings.reload()
        yield
        return

    monkeypatch.setenv("E4EFS_LABEL_STUDIO__URL", "http://label-studio.example.com")
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__API_KEY", "unused")
    monkeypatch.setenv(
        "E4EFS_LABEL_STUDIO__IMAGE_URL_BASE",
        "http://orchestrator.example.com",
    )
    monkeypatch.setenv(
        "E4EFS_FILE_EXCHANGE__URL",
        "http://static-file-server.example.com",
    )
    yield


@pytest.fixture
def label_studio_test_project(request: pytest.FixtureRequest):
    """Create a fresh LS project per integration test, delete on teardown.

    Returns `(project_id, ls_client)`. Title carries a UUID so parallel
    tests don't collide and a leftover from a previously-failed run
    can't conflict with a fresh attempt. The labeling-config XML is a
    minimal `<View><Image/></View>` — enough for LS to accept tasks
    referencing an image URL, which is all the populate path needs.
    """
    if not _is_integration_test(request):
        pytest.skip("label_studio_test_project fixture is integration-only")

    ls: LabelStudio = get_ls_client()

    # LS caps title at 50 chars — keep this short.
    title = f"fs-int-{uuid.uuid4().hex[:8]}"
    project = ls.projects.create(
        title=title,
        label_config="<View><Image name='image' value='$image'/></View>",
    )

    yield project.id, ls

    try:
        ls.projects.delete(project.id)
    except Exception:  # pylint: disable=broad-except
        # Best-effort cleanup. A leftover project just leaks until the
        # next `docker compose down -v` resets the LS volume; not worth
        # failing the test over.
        pass

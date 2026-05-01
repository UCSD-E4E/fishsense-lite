"""Shared fixtures for api-workflow-worker tests.

Dynaconf eagerly validates every `Validator` on first attribute access of
`settings`, not lazily per-setting (see CLAUDE.md). Any test that imports
an activity module — even one that never calls into `get_fs_client` or
`get_ls_client` — risks tripping this if the worker process happens to
read settings during import. The fixture seeds placeholder values for
every required setting so unrelated validators don't reject the test
process.
"""

import pytest


@pytest.fixture(autouse=True)
def configure_worker_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("E4EFS_TEMPORAL__HOST", "temporal")
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__URL", "http://label-studio.example.com")
    monkeypatch.setenv("E4EFS_LABEL_STUDIO__API_KEY", "unused")
    monkeypatch.setenv("E4EFS_E4E_NAS__URL", "http://nas.example.com")
    monkeypatch.setenv("E4EFS_E4E_NAS__USERNAME", "unused")
    monkeypatch.setenv("E4EFS_E4E_NAS__PASSWORD", "unused")
    monkeypatch.setenv("E4EFS_FISHSENSE_API__URL", "http://fishsense-api.example.com")
    yield

"""`E4EFS_DOCKER` decides where every service reads config and writes logs.

The footgun, spelled out in CLAUDE.md: `bool(os.environ.get("E4EFS_DOCKER"))`
treats *any* non-empty string as true, so `E4EFS_DOCKER=false` reads as Docker
mode and sends paths to `/e4efs/*` on a machine that has no such directories.
`deploy/compose.local.yml` sets exactly that value on the devcontainer, so this
is not hypothetical.

It has to be a module-level constant (services import `IS_DOCKER` directly and
Dynaconf reads `get_config_path()` at import time), so the environment is only
consulted once — which is why these tests reload the module rather than
monkeypatching an attribute.
"""

from __future__ import annotations

import importlib

import pytest


def _reload_with(monkeypatch, value: str | None):
    """Re-import `fishsense_shared.config` with `E4EFS_DOCKER` set to `value`."""
    from fishsense_shared import config  # pylint: disable=import-outside-toplevel

    if value is None:
        monkeypatch.delenv("E4EFS_DOCKER", raising=False)
    else:
        monkeypatch.setenv("E4EFS_DOCKER", value)
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore_module():
    """Leave the module as the rest of the suite found it."""
    yield
    from fishsense_shared import config  # pylint: disable=import-outside-toplevel

    importlib.reload(config)


# ── the values that must read as Docker mode ──────────────────────────


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "on", " true "])
def test_explicitly_truthy_values_enable_docker_mode(monkeypatch, value):
    config = _reload_with(monkeypatch, value)

    assert config.IS_DOCKER is True
    assert config.get_config_path() == config.Path("/e4efs/config")
    assert config.get_log_path("svc") == config.Path("/e4efs/logs")


# ── and the ones that must NOT ────────────────────────────────────────


@pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "off", ""])
def test_explicitly_falsy_values_do_not_enable_docker_mode(monkeypatch, value):
    """The regression this file exists for.

    `E4EFS_DOCKER=false` is a real configuration — `deploy/compose.local.yml`
    sets it on the devcontainer — and it must mean "not Docker". Under
    `bool(os.environ.get(...))` every one of these reads as True and config
    resolution jumps to `/e4efs/config`, which doesn't exist outside an image.
    """
    config = _reload_with(monkeypatch, value)

    assert config.IS_DOCKER is False
    assert config.get_config_path() == config.Path(".")


def test_unset_is_not_docker_mode(monkeypatch):
    config = _reload_with(monkeypatch, None)

    assert config.IS_DOCKER is False
    assert config.get_config_path() == config.Path(".")


def test_an_unrecognised_value_is_not_docker_mode(monkeypatch):
    """Fail safe. An unexpected value should degrade to the local-dev path,
    which is merely wrong, rather than to `/e4efs/*`, which doesn't exist and
    fails at import."""
    config = _reload_with(monkeypatch, "maybe")

    assert config.IS_DOCKER is False


def test_the_compose_local_devcontainer_value_is_honoured(monkeypatch):
    """Pins the concrete consumer: `deploy/compose.local.yml` sets
    `E4EFS_DOCKER: "false"` on the dev service and expects cwd-relative
    config."""
    config = _reload_with(monkeypatch, "false")

    assert config.get_config_path() == config.Path(".")

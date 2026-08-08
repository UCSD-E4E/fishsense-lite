"""Coverage for the remaining `fishsense_shared` surface: log/config paths,
the dynaconf validators, and the standard logging setup.

None of this is glamorous, but all of it runs at import time in every service,
so a break here takes down all four workers at once rather than degrading one
feature. `configure_logging` in particular runs before any of them can log
about the fact that it didn't work.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from pathlib import Path

import pytest

from fishsense_shared import config as config_module
from fishsense_shared import logging as logging_module


# ── get_log_path ──────────────────────────────────────────────────────


def test_log_path_outside_docker_uses_platformdirs_and_creates_it(monkeypatch, tmp_path):
    """The non-Docker branch has a side effect — it creates the directory —
    which is what makes `configure_logging` able to open a file handler on a
    dev box that has never run the service before."""
    monkeypatch.setattr(config_module, "IS_DOCKER", False)

    target = tmp_path / "nested" / "logs"

    class _Dirs:  # pylint: disable=too-few-public-methods
        user_log_path = target

    monkeypatch.setattr(config_module.platformdirs, "PlatformDirs", lambda _: _Dirs())

    result = config_module.get_log_path("svc")

    assert result == target
    assert target.is_dir()


def test_log_path_creation_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "IS_DOCKER", False)
    target = tmp_path / "logs"
    target.mkdir()

    class _Dirs:  # pylint: disable=too-few-public-methods
        user_log_path = target

    monkeypatch.setattr(config_module.platformdirs, "PlatformDirs", lambda _: _Dirs())

    assert config_module.get_log_path("svc") == target   # must not raise


def test_log_path_in_docker_does_not_touch_the_filesystem(monkeypatch):
    """`/e4efs/logs` is a mounted volume owned by the image; creating it here
    would mask a missing mount."""
    monkeypatch.setattr(config_module, "IS_DOCKER", True)

    assert config_module.get_log_path("svc") == Path("/e4efs/logs")


# ── validators ────────────────────────────────────────────────────────


def test_path_validator_accepts_an_existing_path(tmp_path):
    assert config_module.path_validator(str(tmp_path)) is True


def test_path_validator_rejects_a_missing_path(tmp_path):
    assert config_module.path_validator(str(tmp_path / "nope")) is False


@pytest.mark.parametrize(
    "url",
    [
        "http://fishsense-api:8000",     # docker DNS, no TLD
        "http://static_file_server",     # underscore
        "https://api.fishsense.e4e.ucsd.edu",
        "http://garage:3900",
    ],
)
def test_url_condition_accepts_the_hostnames_this_repo_actually_uses(url):
    """`validators.url` rejects every one of these, which is why this exists.
    See the docstring in config.py — don't switch back."""
    assert config_module.url_condition(url) is True


@pytest.mark.parametrize(
    "value",
    ["fishsense-api:8000", "ftp://host/x", "", "://nohost", "http://"],
)
def test_url_condition_still_rejects_malformed_urls(value):
    assert config_module.url_condition(value) is False


@pytest.mark.parametrize("value", [None, 8000, b"http://x", ["http://x"]])
def test_url_condition_rejects_non_strings(value):
    """Dynaconf will hand over whatever is in the settings file, so a
    non-string must return False rather than raising inside validation."""
    assert config_module.url_condition(value) is False


# ── configure_logging ─────────────────────────────────────────────────


@pytest.fixture
def clean_root_logger():
    """`configure_logging` mutates the root logger and the global Formatter
    converter. Restore both, or every later test inherits the handlers."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    saved_converter = logging.Formatter.converter
    root.handlers = []
    yield root
    for handler in root.handlers:
        handler.close()
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    logging.Formatter.converter = saved_converter


def test_configure_logging_installs_a_rotating_file_handler_and_a_console_handler(
    monkeypatch, tmp_path, clean_root_logger
):
    monkeypatch.setattr(logging_module, "get_log_path", lambda _: tmp_path)

    logging_module.configure_logging("svc")

    kinds = [type(h) for h in clean_root_logger.handlers]
    assert logging.handlers.TimedRotatingFileHandler in kinds
    assert logging.StreamHandler in kinds
    assert clean_root_logger.level == logging.DEBUG


@pytest.mark.usefixtures("clean_root_logger")
def test_configure_logging_defaults_the_filename_to_the_app_name(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_module, "get_log_path", lambda _: tmp_path)

    logging_module.configure_logging("my-worker")

    assert (tmp_path / "my-worker.log").exists()


@pytest.mark.usefixtures("clean_root_logger")
def test_configure_logging_honours_an_explicit_filename(monkeypatch, tmp_path):
    monkeypatch.setattr(logging_module, "get_log_path", lambda _: tmp_path)

    logging_module.configure_logging("my-worker", log_filename="custom.log")

    assert (tmp_path / "custom.log").exists()
    assert not (tmp_path / "my-worker.log").exists()


@pytest.mark.usefixtures("clean_root_logger")
def test_configure_logging_switches_timestamps_to_utc(monkeypatch, tmp_path):
    """Every service's logs are read side by side against Temporal's UTC
    timestamps; a local-time handler makes correlating an incident guesswork."""
    monkeypatch.setattr(logging_module, "get_log_path", lambda _: tmp_path)

    logging_module.configure_logging("svc")

    assert logging.Formatter.converter is time.gmtime


def test_configure_log_handler_sets_debug_and_the_shared_format():
    handler = logging.StreamHandler()

    logging_module.configure_log_handler(handler)

    assert handler.level == logging.DEBUG
    assert handler.formatter is not None
    # Millisecond precision + trailing Z — the format the log-shipping side
    # parses.
    assert ".%(msecs)03dZ" in handler.formatter._fmt  # pylint: disable=protected-access


def test_log_lines_render_in_the_expected_shape(
    monkeypatch, tmp_path, clean_root_logger
):
    """End to end: a real record through the real handler onto disk."""
    monkeypatch.setattr(logging_module, "get_log_path", lambda _: tmp_path)
    logging_module.configure_logging("svc")

    logging.getLogger("some.module").warning("hello %s", "world")
    for handler in clean_root_logger.handlers:
        handler.flush()

    written = (tmp_path / "svc.log").read_text(encoding="utf-8")
    assert "some.module - WARNING - hello world" in written
    assert "Z - " in written          # UTC marker survived formatting

# pylint: disable=protected-access
"""Unit tests for NasBackupClient.

Pins the regression for the 2026-05-03 backup-worker incident:
synology-api 0.8.x caches `Authentication` on
`BaseApi.shared_session` (class-level), so successive
`FileStation()` constructors reuse the same in-memory SID across
activity attempts. After DSM idle-times-out the session, every
subsequent call returns "Invalid session / SID not found".
NasBackupClient.__init__ must reset that class attribute before
constructing FileStation so each client gets a real login.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from synology_api.base_api import BaseApi


def test_init_resets_shared_session_before_constructing_filestation(monkeypatch):
    """Regression: a stale `BaseApi.shared_session` from a previous
    activity attempt must be cleared so the new FileStation()
    constructor logs in fresh, rather than reusing an SID DSM has
    already idle-timed-out.
    """
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    # Pretend a previous activity attempt left a session on the class.
    BaseApi.shared_session = MagicMock(name="stale-prior-session")

    seen_shared_session: list = []

    def fake_filestation(*_args, **_kwargs):
        # Capture what `BaseApi.shared_session` looks like at the
        # moment FileStation() runs — that's the contract we care
        # about. By the real synology-api lib's logic this is what
        # determines whether login() runs or the stale session is
        # reused.
        seen_shared_session.append(BaseApi.shared_session)
        return MagicMock(name="fs-client")

    monkeypatch.setattr(sut, "FileStation", fake_filestation)

    sut.NasBackupClient(
        nas_url="https://nas.example.com:6021",
        username="u",
        password="p",
    )

    assert seen_shared_session == [None], (
        "FileStation was constructed while BaseApi.shared_session was "
        f"still set to {seen_shared_session[0]!r}; the stale session "
        "would be reused and DSM would reject calls after idle timeout."
    )


def test_init_resets_shared_session_on_each_construction(monkeypatch):
    """Same contract repeated across constructions — every new client
    gets a clean class state, regardless of what the previous one
    left behind."""
    from fishsense_backup_worker import nas as sut  # pylint: disable=import-outside-toplevel

    seen_shared_session: list = []

    def fake_filestation(*_args, **_kwargs):
        seen_shared_session.append(BaseApi.shared_session)
        # Simulate the real lib's behavior: when FileStation logs in,
        # it stores its Authentication on the class attribute.
        BaseApi.shared_session = MagicMock(name="fresh-session")
        return MagicMock(name="fs-client")

    monkeypatch.setattr(sut, "FileStation", fake_filestation)

    for _ in range(3):
        sut.NasBackupClient(
            nas_url="https://nas.example.com:6021", username="u", password="p"
        )

    assert seen_shared_session == [None, None, None]


@pytest.fixture(autouse=True)
def _restore_shared_session():
    """Tests touch a module-global on synology-api; restore between runs
    so an unrelated test that imports this module isn't affected."""
    saved = BaseApi.shared_session
    yield
    BaseApi.shared_session = saved

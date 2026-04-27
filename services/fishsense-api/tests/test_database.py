from types import SimpleNamespace

import pytest

from fishsense_api import config, database


@pytest.fixture
def fake_settings(monkeypatch):
    fake_pg = SimpleNamespace(
        username="u", password="p", host="localhost", port=5432, database="d"
    )
    monkeypatch.setattr(config, "settings", SimpleNamespace(postgres=fake_pg))


@pytest.fixture
def reset_session_factory():
    """Snapshot and restore the module-global session factory around each test."""
    original = database._session_factory
    database._session_factory = None
    yield
    database._session_factory = original


def test_setup_database_returns_database_and_populates_factory(
    fake_settings, reset_session_factory
):
    assert database._session_factory is None

    db = database.setup_database()

    assert isinstance(db, database.Database)
    assert database._session_factory is not None


async def test_get_async_session_raises_before_setup(reset_session_factory):
    """Regression: get_async_session must surface a clear error if the lifespan
    hasn't run yet, rather than dereferencing None."""
    with pytest.raises(RuntimeError, match="setup_database"):
        async for _ in database.get_async_session():
            pass

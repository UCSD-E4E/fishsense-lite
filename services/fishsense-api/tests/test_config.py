from types import SimpleNamespace

from fishsense_api import config


def test_pg_connection_string_uses_asyncpg_driver_and_format(monkeypatch):
    fake_pg = SimpleNamespace(
        username="alice",
        password="s3cr3t",
        host="db.internal",
        port=6543,
        database="fish",
    )
    monkeypatch.setattr(config, "settings", SimpleNamespace(postgres=fake_pg))

    assert (
        config.pg_connection_string()
        == "postgresql+asyncpg://alice:s3cr3t@db.internal:6543/fish"
    )


def test_pg_connection_string_reads_settings_lazily(monkeypatch):
    """Each call must re-read settings — supports settings reload at runtime."""
    pg_a = SimpleNamespace(
        username="a", password="a", host="h1", port=5432, database="d"
    )
    pg_b = SimpleNamespace(
        username="b", password="b", host="h2", port=5432, database="d"
    )
    monkeypatch.setattr(config, "settings", SimpleNamespace(postgres=pg_a))
    first = config.pg_connection_string()

    monkeypatch.setattr(config, "settings", SimpleNamespace(postgres=pg_b))
    second = config.pg_connection_string()

    assert first != second
    assert "@h1:" in first
    assert "@h2:" in second

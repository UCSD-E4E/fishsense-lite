"""`build_tls_config` is the single mTLS implementation for the whole repo.

Every service that talks to Temporal builds its client identity through this
one function, so a mistake here doesn't degrade a feature — it disconnects all
four workers from krg-prod at once, and only in production, because local dev
runs `temporal server start-dev` with `tls = false` and never reaches the
interesting branches.

It was entirely uncovered until this file.
"""

from __future__ import annotations

import pytest


class _Settings:
    """Stand-in for the `settings.temporal` subtree.

    Dynaconf boxes support both attribute access and `in`, and
    `build_tls_config` uses both (`temporal_settings.tls` /
    `"domain" in temporal_settings`), so the double has to as well —
    a plain namespace would pass attribute access and silently fail
    the membership tests.
    """

    def __init__(self, **values):
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def __contains__(self, key):
        return key in self._values


@pytest.fixture
def certs(tmp_path):
    client_cert = tmp_path / "client.crt"
    client_key = tmp_path / "client.key"
    root_ca = tmp_path / "ca.crt"
    client_cert.write_bytes(b"-----BEGIN CERTIFICATE-----client")
    client_key.write_bytes(b"-----BEGIN PRIVATE KEY-----key")
    root_ca.write_bytes(b"-----BEGIN CERTIFICATE-----ca")
    return client_cert, client_key, root_ca


def test_tls_disabled_returns_none_so_local_dev_connects_plaintext():
    """`temporal server start-dev` has no TLS; returning None is what lets the
    same call site work in both environments."""
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    assert build_tls_config(_Settings(tls=False)) is None


def test_reads_the_cert_files_from_disk_as_bytes(certs):
    """The certs are paths in settings but bytes on the wire — passing the path
    through would produce a TLSConfig that fails at connect time, far from
    here."""
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    client_cert, client_key, _ = certs
    result = build_tls_config(
        _Settings(
            tls=True,
            client_cert=str(client_cert),
            client_private_key=str(client_key),
        )
    )

    assert result is not None
    assert result.client_cert == b"-----BEGIN CERTIFICATE-----client"
    assert result.client_private_key == b"-----BEGIN PRIVATE KEY-----key"


def test_server_root_ca_is_read_when_present(certs):
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    client_cert, client_key, root_ca = certs
    result = build_tls_config(
        _Settings(
            tls=True,
            client_cert=str(client_cert),
            client_private_key=str(client_key),
            server_root_ca_cert=str(root_ca),
        )
    )

    assert result.server_root_ca_cert == b"-----BEGIN CERTIFICATE-----ca"


def test_server_root_ca_is_none_when_absent(certs):
    """Optional: a public CA needs no explicit root. The key thing is that the
    absence is detected by membership, not by reading a missing file."""
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    client_cert, client_key, _ = certs
    result = build_tls_config(
        _Settings(
            tls=True,
            client_cert=str(client_cert),
            client_private_key=str(client_key),
        )
    )

    assert result.server_root_ca_cert is None


def test_domain_is_passed_through_when_present(certs):
    """`domain` is the serverName the client verifies against. krg-prod's cert
    is issued for a name that isn't the connection host, so dropping this
    fails the handshake."""
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    client_cert, client_key, _ = certs
    result = build_tls_config(
        _Settings(
            tls=True,
            client_cert=str(client_cert),
            client_private_key=str(client_key),
            domain="krg-prod.ucsd.edu",
        )
    )

    assert result.domain == "krg-prod.ucsd.edu"


def test_domain_is_none_when_absent(certs):
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    client_cert, client_key, _ = certs
    result = build_tls_config(
        _Settings(
            tls=True,
            client_cert=str(client_cert),
            client_private_key=str(client_key),
        )
    )

    assert result.domain is None


def test_a_missing_cert_file_raises_rather_than_connecting_without_tls(tmp_path):
    """Fail loud. Silently degrading to no client identity would connect to
    Temporal unauthenticated (or not at all) with nothing pointing at the
    missing file."""
    from fishsense_shared.temporal import (  # pylint: disable=import-outside-toplevel
        build_tls_config,
    )

    with pytest.raises(FileNotFoundError):
        build_tls_config(
            _Settings(
                tls=True,
                client_cert=str(tmp_path / "absent.crt"),
                client_private_key=str(tmp_path / "absent.key"),
            )
        )

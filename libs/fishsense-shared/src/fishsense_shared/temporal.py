"""Build a Temporal mTLS config from Dynaconf settings.

Expects the standard ``settings.temporal`` shape used by every FishSense
worker: ``tls`` (bool), ``client_cert``, ``client_private_key``, optional
``server_root_ca_cert`` and ``domain``. Returns ``None`` when ``tls`` is False
so the caller can pass it straight to ``Client.connect(..., tls=tls_config)``.
"""

from pathlib import Path

from temporalio.client import TLSConfig


def build_tls_config(temporal_settings) -> TLSConfig | None:
    """Read cert files referenced in ``temporal_settings`` and assemble a TLSConfig."""
    if not temporal_settings.tls:
        return None

    client_cert = Path(temporal_settings.client_cert).read_bytes()
    client_private_key = Path(temporal_settings.client_private_key).read_bytes()

    server_root_ca_cert: bytes | None = None
    if "server_root_ca_cert" in temporal_settings:
        server_root_ca_cert = Path(temporal_settings.server_root_ca_cert).read_bytes()

    return TLSConfig(
        client_cert=client_cert,
        client_private_key=client_private_key,
        server_root_ca_cert=server_root_ca_cert,
        domain=temporal_settings.domain if "domain" in temporal_settings else None,
    )

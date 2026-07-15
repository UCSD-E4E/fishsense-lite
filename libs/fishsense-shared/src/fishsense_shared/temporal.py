"""Build a Temporal mTLS config from Dynaconf settings.

Expects the standard ``settings.temporal`` shape used by every FishSense
worker: ``tls`` (bool), ``client_cert``, ``client_private_key``, optional
``server_root_ca_cert`` and ``domain``. Returns ``None`` when ``tls`` is False
so the caller can pass it straight to ``Client.connect(..., tls=tls_config)``.

Also exposes ``ensure_schedule`` — every worker that owns a recurring
schedule registers it idempotently at startup, refusing to update an
existing schedule in-place. Operators must `temporal schedule delete`
and let the next worker startup recreate it; this prevents config typos
from silently retiring or mutating production schedules.
"""

import logging
from pathlib import Path

from temporalio.client import (
    Client,
    Schedule,
    ScheduleAlreadyRunningError,
    TLSConfig,
)

_log = logging.getLogger(__name__)


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


def temporal_namespace(temporal_settings) -> str:
    """Return the configured Temporal namespace, defaulting to ``default``.

    OSS Temporal mTLS authenticates the client but does NOT pin it to a
    namespace (krg-infra ADR 0023 — "namespace isolation is by convention,
    not enforced"). A worker that omits ``namespace=`` on ``Client.connect``
    silently lands in ``default`` instead of the tenant's ``fishsense``
    namespace, so every worker threads this in. Tolerant of an unset key the
    same way ``build_tls_config`` is for ``domain`` — falls back rather than
    raising, so local dev / tests that don't set it match temporalio's own
    ``default``.
    """
    if "namespace" in temporal_settings:
        return temporal_settings.namespace
    return "default"


async def ensure_schedule(
    client: Client,
    *,
    schedule_id: str,
    schedule: Schedule,
) -> None:
    """Create the schedule if it doesn't exist; otherwise no-op.

    Workers call this at startup. We deliberately do NOT update existing
    schedules — operators have to delete + redeploy if they want config
    changes to take effect, so a config typo can't silently retire or
    mutate a production schedule.
    """
    try:
        await client.create_schedule(schedule_id, schedule)
        _log.info("created schedule %s", schedule_id)
    except ScheduleAlreadyRunningError:
        _log.info(
            "schedule %s already exists; leaving as-is "
            "(delete + redeploy to pick up config changes)",
            schedule_id,
        )

"""Activity: pg_dump one DB to a local tmpfile, upload to NAS, delete
the tmpfile.

The dump-and-upload are intentionally one activity. If we split them,
the tmpfile lives only on the worker pod that ran pg_dump — a retry
on a different pod wouldn't see it. Keeping them together also means
a partial dump never reaches the NAS.

Each invocation gets its own `TemporaryDirectory`, so concurrent
per-DB activities (the workflow fans out across `fishsense`,
`superset`, `temporal_db` in parallel) can't collide on a shared
`/tmp/<timestamp>.dump` filename. The 2026-05-03 incident traced to
the prior shape: all three activities' rename targets resolved to
the same path, the fastest-finishing one's `finally` deleted it,
and the others raised `FileNotFoundError` on upload.
"""

import asyncio
import logging
import os
import tempfile
from datetime import datetime, timezone

from temporalio import activity

from fishsense_backup_worker.activities._heartbeat import heartbeat_pump
from fishsense_backup_worker.backup_naming import backup_filename
from fishsense_backup_worker.config import settings
from fishsense_backup_worker.nas import NasBackupClient
from fishsense_backup_worker.pg_dump import run_pg_dump

_log = logging.getLogger(__name__)


def _input_model():
    # pylint: disable=import-outside-toplevel
    from fishsense_backup_worker.workflows.backup_databases_workflow \
        import PgDumpDatabaseInput

    return PgDumpDatabaseInput


def _dump_and_upload(*, db_name: str, nas_root_path: str) -> str:
    """Sync helper run via asyncio.to_thread. Returns the NAS folder
    path the dump landed in (purely for logging — Temporal doesn't
    need the return value)."""
    filename = backup_filename(datetime.now(tz=timezone.utc))
    nas_dir = f"{nas_root_path.rstrip('/')}/{db_name}"

    with tempfile.TemporaryDirectory(prefix=f"backup-{db_name}-") as tmpdir:
        # The local path's basename is what `synology-api`'s
        # `upload_file` preserves on the NAS, so write directly to
        # the canonical filename inside our isolated tempdir — no
        # rename, no shared `/tmp` path collision possible.
        local_path = os.path.join(tmpdir, filename)

        run_pg_dump(
            db_name=db_name,
            host=settings.postgres.host,
            port=int(settings.postgres.port),
            username=settings.postgres.username,
            password=settings.postgres.password,
            output_path=local_path,
        )

        nas = NasBackupClient(
            nas_url=settings.e4e_nas.url,
            username=settings.e4e_nas.username,
            password=settings.e4e_nas.password,
        )
        nas.upload(dest_dir=nas_dir, src_file_path=local_path)
        return nas_dir


@activity.defn
async def pg_dump_database(payload) -> None:  # type: ignore[no-untyped-def]
    payload_cls = _input_model()
    if not isinstance(payload, payload_cls):
        payload = payload_cls.model_validate(payload)

    activity.logger.info(
        "pg_dump_database start db=%s nas_root=%s",
        payload.db_name,
        payload.nas_root_path,
    )

    async with heartbeat_pump():
        await asyncio.to_thread(
            _dump_and_upload,
            db_name=payload.db_name,
            nas_root_path=payload.nas_root_path,
        )

    activity.logger.info("pg_dump_database done db=%s", payload.db_name)

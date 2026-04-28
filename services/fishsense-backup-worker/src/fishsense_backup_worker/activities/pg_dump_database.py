"""Activity: pg_dump one DB to a local tmpfile, upload to NAS, delete
the tmpfile.

The dump-and-upload are intentionally one activity. If we split them,
the tmpfile lives only on the worker pod that ran pg_dump — a retry
on a different pod wouldn't see it. Keeping them together also means
a partial dump never reaches the NAS.
"""

import asyncio
import logging
import os
import tempfile

from temporalio import activity

from fishsense_backup_worker.backup_naming import backup_filename
from fishsense_backup_worker.config import settings
from fishsense_backup_worker.nas import NasBackupClient
from fishsense_backup_worker.pg_dump import run_pg_dump

_log = logging.getLogger(__name__)


def _input_model():
    from fishsense_backup_worker.workflows.backup_databases_workflow import (
        PgDumpDatabaseInput,
    )

    return PgDumpDatabaseInput


def _dump_and_upload(*, db_name: str, nas_root_path: str) -> str:
    """Sync helper run via asyncio.to_thread. Returns the NAS folder
    path the dump landed in (purely for logging — Temporal doesn't
    need the return value)."""
    from datetime import datetime, timezone

    filename = backup_filename(datetime.now(tz=timezone.utc))
    nas_dir = f"{nas_root_path.rstrip('/')}/{db_name}"

    # tempfile.NamedTemporaryFile with delete=False so we can pass the
    # path to pg_dump (which won't reuse our open handle); we delete
    # in finally.
    with tempfile.NamedTemporaryFile(
        prefix=f"{db_name}-", suffix=".dump", delete=False
    ) as tmp:
        local_path = tmp.name

    try:
        run_pg_dump(
            db_name=db_name,
            host=settings.postgres.host,
            port=int(settings.postgres.port),
            username=settings.postgres.username,
            password=settings.postgres.password,
            output_path=local_path,
        )

        # Rename the tmpfile to the final filename so the NAS upload
        # carries the canonical name (synology-api's upload preserves
        # the local file's basename).
        renamed = os.path.join(os.path.dirname(local_path), filename)
        os.replace(local_path, renamed)
        local_path = renamed

        nas = NasBackupClient(
            nas_url=settings.e4e_nas.url,
            username=settings.e4e_nas.username,
            password=settings.e4e_nas.password,
        )
        nas.upload(dest_dir=nas_dir, src_file_path=local_path)
        return nas_dir
    finally:
        try:
            os.remove(local_path)
        except FileNotFoundError:
            pass


@activity.defn
async def pg_dump_database(input) -> None:  # type: ignore[no-untyped-def]
    PgDumpDatabaseInput = _input_model()
    if not isinstance(input, PgDumpDatabaseInput):
        input = PgDumpDatabaseInput.model_validate(input)

    activity.logger.info(
        "pg_dump_database start db=%s nas_root=%s",
        input.db_name,
        input.nas_root_path,
    )

    await asyncio.to_thread(
        _dump_and_upload,
        db_name=input.db_name,
        nas_root_path=input.nas_root_path,
    )

    activity.logger.info("pg_dump_database done db=%s", input.db_name)

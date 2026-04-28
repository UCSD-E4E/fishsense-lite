"""Activity: list NAS backups for one DB, delete anything beyond the
retention window. Pure pruning logic lives in `backup_naming.py` and
is unit-tested; this module is the I/O wrapper."""

import asyncio
import logging
from typing import List

from temporalio import activity

from fishsense_backup_worker.backup_naming import filenames_to_prune
from fishsense_backup_worker.config import settings
from fishsense_backup_worker.nas import NasBackupClient

_log = logging.getLogger(__name__)


def _input_model():
    from fishsense_backup_worker.workflows.backup_databases_workflow import (
        PruneDatabaseBackupsInput,
    )

    return PruneDatabaseBackupsInput


def _prune(
    *, db_name: str, nas_root_path: str, keep: int
) -> List[str]:
    """Sync helper run via asyncio.to_thread. Returns the list of
    filenames pruned (mostly for logging)."""
    nas = NasBackupClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )
    nas_dir = f"{nas_root_path.rstrip('/')}/{db_name}"
    existing = nas.list_filenames(folder_path=nas_dir)
    to_delete = filenames_to_prune(existing, keep=keep)

    for filename in to_delete:
        nas.delete(file_path=f"{nas_dir}/{filename}")
    return to_delete


@activity.defn
async def prune_database_backups(input) -> None:  # type: ignore[no-untyped-def]
    PruneDatabaseBackupsInput = _input_model()
    if not isinstance(input, PruneDatabaseBackupsInput):
        input = PruneDatabaseBackupsInput.model_validate(input)

    activity.logger.info(
        "prune_database_backups start db=%s keep=%d",
        input.db_name,
        input.keep,
    )

    pruned = await asyncio.to_thread(
        _prune,
        db_name=input.db_name,
        nas_root_path=input.nas_root_path,
        keep=input.keep,
    )

    activity.logger.info(
        "prune_database_backups done db=%s pruned_count=%d",
        input.db_name,
        len(pruned),
    )

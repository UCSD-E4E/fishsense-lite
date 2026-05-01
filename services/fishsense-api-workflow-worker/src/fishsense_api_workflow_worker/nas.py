"""FileStation-backed NAS client for the api-worker's staging path.

Mirrors the backup-worker's `NasBackupClient` (`services/fishsense-
backup-worker/src/fishsense_backup_worker/nas.py`) but exposes only
the operations the staging activities need: download a single file
to a local path. Sync (synology-api is sync) — call from inside
`asyncio.to_thread` if running from an async activity.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from urllib.parse import urlparse

from synology_api.exceptions import FileStationError
from synology_api.filestation import FileStation

_log = logging.getLogger(__name__)


@contextmanager
def _surface_filestation_error(op: str):
    """synology-api's FileStationError stores the human-readable message
    on `error_message` but never passes it to Exception.__init__, so
    `str(e)` is empty and Temporal failure events show just the class
    name. Catch and re-raise as a regular Exception that carries the
    decoded message + the operation that failed."""
    try:
        yield
    except FileStationError as e:
        msg = getattr(e, "error_message", "") or repr(e)
        _log.error("nas %s failed: %s", op, msg)
        raise RuntimeError(f"FileStation {op} failed: {msg}") from e


class NasDownloadClient:
    """FileStation-backed client for the staging-in path.

    `nas_url` is parsed for hostname + port; `username`/`password` are
    the corresponding NAS credentials. Same parse contract as the
    backup-worker's `NasBackupClient`.
    """

    def __init__(self, *, nas_url: str, username: str, password: str):
        parsed = urlparse(nas_url)
        if not parsed.hostname or not parsed.port:
            raise ValueError(
                f"NAS url must include hostname + port; got {nas_url!r}"
            )
        self._fs = FileStation(
            parsed.hostname,
            parsed.port,
            username,
            password,
            secure=True,
            cert_verify=False,
        )

    def download_to(self, *, src_path: str, dest_dir: str) -> None:
        """Download the NAS file at `src_path` into local `dest_dir`.

        synology-api's `get_file(mode='download', dest_path=...)`
        writes to a local directory using the source filename — so
        the resulting file lands at
        `{dest_dir}/{basename(src_path)}`. Caller is responsible for
        creating `dest_dir` (typically a `tempfile.TemporaryDirectory`).
        """
        _log.info("nas download start src=%s dest_dir=%s", src_path, dest_dir)
        with _surface_filestation_error(f"download {src_path}"):
            self._fs.get_file(
                path=src_path,
                mode="download",
                dest_path=dest_dir,
            )
        _log.info("nas download done src=%s", src_path)

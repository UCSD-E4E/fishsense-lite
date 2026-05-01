"""FileStation-backed NAS client for the api-worker.

Mirrors the backup-worker's `NasBackupClient` (`services/fishsense-
backup-worker/src/fishsense_backup_worker/nas.py`) plus a download
operation. Sync (synology-api is sync) — call from inside
`asyncio.to_thread` if running from an async activity.

Operations:
  * `download_to(src_path, dest_dir)` — Phase 3a staging-in.
  * `upload(dest_dir, src_file_path)` — Phase 3b archive.
  * `exists(file_path)` — idempotency HEAD-equivalent for archive.
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


class NasClient:
    """FileStation-backed client for the api-worker.

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

    def upload(self, *, dest_dir: str, src_file_path: str) -> None:
        """Upload `src_file_path` into NAS folder `dest_dir`.

        Creates `dest_dir` (idempotently) before the upload —
        synology-api's `upload_file(create_parents=True)` does NOT
        actually create missing parents in practice; the upload
        silently no-ops and returns a non-success tuple. Existing
        files at the same name are overwritten.
        """
        _log.info("nas upload start dest=%s src=%s", dest_dir, src_file_path)
        self._ensure_dir(dest_dir)
        with _surface_filestation_error(f"upload {src_file_path} -> {dest_dir}"):
            result = self._fs.upload_file(
                dest_dir, src_file_path, overwrite=True
            )
        if isinstance(result, tuple):
            status, body = result
            raise RuntimeError(
                f"FileStation upload {src_file_path} -> {dest_dir} failed: "
                f"http_status={status} body={body!r}"
            )
        _log.info("nas upload done dest=%s", dest_dir)

    def exists(self, *, file_path: str) -> bool:
        """Return True if `file_path` exists on the NAS, False otherwise.

        Used by the archive activity to skip already-uploaded JPEGs
        on retried runs without re-paying the local read + upload.
        """
        try:
            info = self._fs.get_file_info(path=file_path)
        except FileStationError:
            return False
        # `get_file_info` returns {"data": {"files": [{"name": ..., "isdir": ...}]}}
        # on success and an error-shaped tuple on missing path. Defensive.
        if not isinstance(info, dict):
            return False
        files = info.get("data", {}).get("files", [])
        return any(
            isinstance(f, dict) and f.get("name") for f in files
        )

    def _ensure_dir(self, dest_dir: str) -> None:
        """Create `dest_dir` on the NAS, treating 'already exists' as
        success."""
        parent, _, name = dest_dir.rstrip("/").rpartition("/")
        if not parent or not name:
            return
        try:
            self._fs.create_folder(
                folder_path=parent, name=name, force_parent=True
            )
        except FileStationError as e:
            msg = getattr(e, "error_message", "") or ""
            if "already exists" in msg.lower():
                return
            _log.error("nas ensure_dir %s failed: %s", dest_dir, msg)
            raise RuntimeError(
                f"FileStation create_folder {dest_dir} failed: {msg}"
            ) from e


# Backwards-compatible alias for Phase 3a callers that imported the
# narrow class name. New code should use `NasClient` directly.
NasDownloadClient = NasClient

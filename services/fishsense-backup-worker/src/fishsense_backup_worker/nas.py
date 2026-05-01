"""Thin wrapper around synology-api FileStation for the backup paths
this worker writes to.

Hides the URL parsing + login dance + the awkward upload/list/delete
shape behind three methods this worker actually uses. Every method is
sync (synology-api is sync) — call from inside `asyncio.to_thread` if
running from an async activity.
"""

import logging
from contextlib import contextmanager
from typing import List
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
    code + decoded message + the operation that failed."""
    try:
        yield
    except FileStationError as e:
        msg = getattr(e, "error_message", "") or repr(e)
        _log.error("nas %s failed: %s", op, msg)
        raise RuntimeError(f"FileStation {op} failed: {msg}") from e


class NasBackupClient:
    """FileStation-backed client for the backup workflow's narrow
    needs: upload, list, delete a single file.

    `nas_url` is parsed for hostname + port; `username`/`password` are
    the corresponding NAS credentials.
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

    def upload(self, *, dest_dir: str, src_file_path: str) -> None:
        """Upload `src_file_path` into NAS folder `dest_dir`. Creates
        `dest_dir` (idempotently) before the upload — synology-api's
        `upload_file(create_parents=True)` does NOT actually create
        missing parents in practice, the upload silently no-ops and
        returns a non-success tuple. Existing files at the same name
        are overwritten."""
        _log.info("nas upload start dest=%s src=%s", dest_dir, src_file_path)
        self._ensure_dir(dest_dir)
        with _surface_filestation_error(f"upload {src_file_path} -> {dest_dir}"):
            result = self._fs.upload_file(
                dest_dir, src_file_path, overwrite=True
            )
        # synology-api returns (status_code, json) on app-level failure
        # and a plain dict (or success-shaped json) on success. Anything
        # tuple-shaped is a failure we'd otherwise swallow.
        if isinstance(result, tuple):
            status, body = result
            raise RuntimeError(
                f"FileStation upload {src_file_path} -> {dest_dir} failed: "
                f"http_status={status} body={body!r}"
            )
        _log.info("nas upload done dest=%s", dest_dir)

    def _ensure_dir(self, dest_dir: str) -> None:
        """Create `dest_dir` on the NAS, treating 'already exists' as
        success. Caller passes a single absolute path; we split it into
        parent + name for the FileStation API."""
        parent, _, name = dest_dir.rstrip("/").rpartition("/")
        if not parent or not name:
            # `dest_dir` is the share root itself — nothing to create.
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

    def list_filenames(self, *, folder_path: str) -> List[str]:
        """Return just the bare filenames in `folder_path` (no paths,
        no metadata)."""
        with _surface_filestation_error(f"list {folder_path}"):
            listing = self._fs.get_file_list(folder_path)
        # synology-api returns a dict like {"data": {"files": [{"name": ...}, ...]}}
        # — pluck the names defensively in case the shape changes between versions.
        files = (
            listing.get("data", {}).get("files", []) if isinstance(listing, dict) else []
        )
        return [f["name"] for f in files if isinstance(f, dict) and "name" in f]

    def delete(self, *, file_path: str) -> None:
        """Delete a single file at the given absolute NAS path."""
        _log.info("nas delete %s", file_path)
        with _surface_filestation_error(f"delete {file_path}"):
            self._fs.delete_blocking_function(file_path)

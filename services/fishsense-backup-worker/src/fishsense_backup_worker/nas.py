"""Thin wrapper around synology-api FileStation for the backup paths
this worker writes to.

Hides the URL parsing + login dance + the awkward upload/list/delete
shape behind three methods this worker actually uses. Every method is
sync (synology-api is sync) — call from inside `asyncio.to_thread` if
running from an async activity.
"""

import logging
from typing import List
from urllib.parse import urlparse

from synology_api.filestation import FileStation

_log = logging.getLogger(__name__)


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
        """Upload `src_file_path` into NAS folder `dest_dir`. Folder is
        created if missing (`create_parents=True` on synology-api's
        upload). Existing files at the same name are overwritten."""
        _log.info("nas upload start dest=%s src=%s", dest_dir, src_file_path)
        self._fs.upload_file(dest_dir, src_file_path, overwrite=True)
        _log.info("nas upload done dest=%s", dest_dir)

    def list_filenames(self, *, folder_path: str) -> List[str]:
        """Return just the bare filenames in `folder_path` (no paths,
        no metadata)."""
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
        self._fs.delete_blocking_function(file_path)

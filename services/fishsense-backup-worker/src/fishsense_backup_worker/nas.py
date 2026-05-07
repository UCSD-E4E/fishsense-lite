"""FileStation-backed NAS client for the backup worker.

Backed by the `synology-filestation` Rust client (PyO3 wheel from
`UCSD-E4E/synology-filestation`). Replaces the previous `synology-api`
PyPI package — see the api-worker's `nas.py` for the 2026-05-07 stage
2 incident that drove the swap. The backup worker uses a narrow
slice of the FileStation surface (upload, list, delete), but the
underlying-library footguns that wedged stage 2 (silent JSON-error
returns, no atomic-rename) applied here too — fixing both at once.

Operations:
  * `upload(dest_dir, src_file_path)` — push a fresh `pg_dump` to NAS.
  * `list_filenames(folder_path)` — enumerate existing backups for a DB.
  * `delete(file_path)` — remove a single old backup during pruning.

Sync only — call from `asyncio.to_thread` if running from an async
activity (matches the previous client's contract).
"""

from __future__ import annotations

import logging
import os
from typing import List
from urllib.parse import urlparse

from synology_filestation import (
    AlreadyExists,
    Client,
)

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
        # `auto_relogin=True` (default) makes the client transparently
        # re-authenticate on SID-expired (DSM code 119) — the property
        # that fixes the 30-min idle-timeout pattern that bit the
        # 2026-05-03 backup-worker incident and the 2026-05-07 api-worker
        # stage 2 incident.
        self._fs = Client.login(
            parsed.hostname,
            parsed.port,
            username,
            password,
            https=True,
        )

    def upload(self, *, dest_dir: str, src_file_path: str) -> None:
        """Upload `src_file_path` into NAS folder `dest_dir`. Creates
        `dest_dir` (idempotently) before the upload — the new client
        creates parents by default in `upload`, but historical prod
        backup paths have at least one layout where the parent of the
        target dir doesn't exist, so the explicit `_ensure_dir` step
        stays as belt-and-suspenders. Existing files at the same name
        are overwritten."""
        _log.info("nas upload start dest=%s src=%s", dest_dir, src_file_path)
        self._ensure_dir(dest_dir)
        self._fs.upload(src_file_path, dest_dir, overwrite=True)
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
            self._fs.create_folder(parent, name, _force_parent=True)
        except AlreadyExists:
            return

    def list_filenames(self, *, folder_path: str) -> List[str]:
        """Return just the bare filenames in `folder_path` (no paths,
        no metadata). The Rust client's `list_dir` returns dicts with
        full paths in `path`; strip to basename so the prune activity's
        `backup_naming.filenames_to_prune` (which string-matches on
        filenames) gets the shape it expects."""
        entries = self._fs.list_dir(folder_path)
        names = []
        for entry in entries:
            full_path = entry.get("path") if isinstance(entry, dict) else None
            if not full_path:
                continue
            names.append(os.path.basename(full_path))
        return names

    def delete(self, *, file_path: str) -> None:
        """Delete a single file at the given absolute NAS path."""
        _log.info("nas delete %s", file_path)
        self._fs.delete(file_path)

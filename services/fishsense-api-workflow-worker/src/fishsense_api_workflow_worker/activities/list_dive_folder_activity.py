"""Activity: enumerate the `.ORF` frames in one dive folder on the NAS.

The first step of ingest, and the one that fixes the shape of everything after
it: **one request means one dive, and a dive is exactly one directory.**

That is precedent, not preference. The retired spider crawler walked the tree
recursively but assigned `dive = image.parent.relative_to(data_root)`
(`discovery.py:238`), so a nested folder always became its own dive row and
never extra frames on its parent. Every one of the ~479 dive rows in prod
follows that convention. Recursing here and attaching a subdirectory's frames
to the named dive would merge dives that are distinct rows today — silently,
and with nothing in the data afterwards to say which frames came from where.

So a subdirectory holding `.ORF`s is *reported*, not ingested. That is the
Olympus rollover case: the TG-6 wraps its frame counter at `PA199999` and
starts a child folder mid-dive, which reads as "more frames of this dive" to a
person and as "a second dive" to every convention in the database. Surfacing it
as "here is another dive to submit" keeps the operator deciding.

NAS access here is read-only, in line with the rest of the api-worker: this
activity lists, and nothing else.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

from synology_filestation import DSMError
from temporalio import activity

from fishsense_api_workflow_worker.activities.nas_errors import (
    raise_if_permanent_dsm_error,
)
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.nas import NasDownloadClient, NasEntry
from fishsense_shared.ingest_contracts import IngestDiveRequest, SubfolderReport

# Case-insensitive: Olympus writes `.ORF`, but operators and copy tools produce
# `.orf` and occasionally `.Orf`. A case-sensitive match would ingest a partial
# dive that no later step can detect as partial, because the missing frames
# were never listed in the first place.
_RAW_SUFFIX = ".orf"

__all__ = [
    "DiveFolderListing",
    "list_dive_folder_activity",
    "resolve_nas_folder",
]


@dataclass
class DiveFolderListing:
    """What is in the folder, before anything has been read or written."""

    #: The absolute NAS path actually listed, after root resolution.
    folder_path: str
    #: `.ORF` files directly inside `folder_path`, name-sorted.
    files: List[NasEntry] = field(default_factory=list)
    #: Immediate subdirectories that hold `.ORF`s — separate dives, reported
    #: for the operator to submit themselves.
    subfolders: List[SubfolderReport] = field(default_factory=list)


def _build_nas_client() -> NasDownloadClient:
    return NasDownloadClient(
        nas_url=settings.e4e_nas.url,
        username=settings.e4e_nas.username,
        password=settings.e4e_nas.password,
    )


def resolve_nas_folder(relative_path: str) -> str:
    """Join `e4e_nas.raw_root_path` with a share-relative request path.

    The DB stores paths share-relative while FileStation needs them absolute —
    the same asymmetry `stage_raw_bytes_for_dive_activity` handles, and worth
    getting right because FileStation surfaces an unresolved path as a 502
    rather than a 404. An already-absolute path passes through, so an operator
    pasting a full NAS path isn't double-prefixed.
    """
    if relative_path.startswith("/"):
        return relative_path.rstrip("/")
    root = settings.e4e_nas.raw_root_path.rstrip("/")
    return f"{root}/{relative_path.strip('/')}"


def _is_raw(entry: NasEntry) -> bool:
    return not entry.is_dir and entry.name.lower().endswith(_RAW_SUFFIX)


async def _list(client, folder_path: str) -> List[NasEntry]:
    """One `list_dir`, with permanent errors classified.

    No inner retry loop: the bounded jittered Temporal policy owns backoff, and
    an inner loop underneath it is what produced the download storm that
    tripped the NAS auto-block.
    """
    try:
        return await asyncio.to_thread(client.list_dir, folder_path=folder_path)
    except DSMError as exc:
        raise_if_permanent_dsm_error(exc, context=folder_path)
        raise


@activity.defn
async def list_dive_folder_activity(
    request: IngestDiveRequest,
) -> DiveFolderListing:
    client = _build_nas_client()
    folder_path = resolve_nas_folder(request.dive_path)

    entries = await _list(client, folder_path)

    # Name-sorted, because batching and heartbeat-resume both index into this
    # list. DSM promises no order, so a retry that saw a different one would
    # re-download frames it had already registered and skip ones it hadn't.
    files = sorted((e for e in entries if _is_raw(e)), key=lambda e: e.name)

    subfolders: List[SubfolderReport] = []
    for entry in sorted(
        (e for e in entries if e.is_dir), key=lambda e: e.name
    ):
        # Exactly one level down: enough to count a rollover folder's frames,
        # and no more. A full walk would turn a path mistyped near the share
        # root into an enumeration of the entire NAS, over a download backend
        # that already falls over under load.
        children = await _list(client, entry.path)
        orf_count = sum(1 for child in children if _is_raw(child))
        if orf_count:
            subfolders.append(
                SubfolderReport(path=entry.path, orf_count=orf_count)
            )

    activity.logger.info(
        "listed dive folder path=%s frames=%d subfolders=%d",
        folder_path,
        len(files),
        len(subfolders),
    )
    return DiveFolderListing(
        folder_path=folder_path, files=files, subfolders=subfolders
    )

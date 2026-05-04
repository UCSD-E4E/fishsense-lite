"""Read-only scan for Image rows whose `path` 404s on the NAS, plus a
guess at the Olympus rollover-corrected path.

Background
----------
The Olympus TG-6 wraps its frame counter at PA199999. When that
happens mid-dive, subsequent frames land in a child directory whose
name inserts a digit before the `_<rig>` suffix on the original dive
folder. Example:

    recorded:  .../101923_Alligator_FSL06/PA190199.ORF        (404)
    actual:    .../101923_Alligator_FSL06/101923_Alligator1_FSL06/PA190199.ORF

Reported by a downstream consumer who hit it on dives 237 and 249
(303 affected rows total). The image-ingest path that wrote those
`Image.path` values does not live in this monorepo, so this script
only diagnoses — it never mutates the API or NAS.

What it does
------------
For each dive in scope:
  1. Fetch images via fishsense-api-sdk.
  2. HEAD-equivalent each `image.path` against FileStation
     (`get_file_info`).
  3. For every 404, walk Olympus rollover candidates
     (`<basename>1_<rig>/`, `<basename>2_<rig>/`, ...) up to
     `--max-rollover-depth` and report the first one that resolves.
  4. Emit a summary table + a JSON file with the (dive_id, image_id,
     image_path, image_checksum, suggested_rollover_path) rows the
     downstream user asked for.

What it does NOT do
-------------------
- No PUT/POST against fishsense-api. No DB writes. No NAS writes.
- No fix is applied — the script's output is the patch list, not the
  patch.
- No assumption that "contiguous tail of 404s" is the only valid
  symptom. The script reports per-image resolution status; the
  contiguous-tail check is offered as a `--require-contiguous-tail`
  flag for callers who want to filter out random missing files.

Run
---
From the repo root, inside the api-workflow-worker venv (which has
both `fishsense-api-sdk` and `synology-api` installed):

    uv run --package fishsense-api-workflow-worker \\
        python tools/scan_image_path_rollover.py \\
        --api-url https://orchestrator.fishsense.e4e.ucsd.edu \\
        --api-user '<user>' --api-pass '<pass>' \\
        --nas-url https://nas.example:5001 \\
        --nas-user '<user>' --nas-pass '<pass>' \\
        --nas-root /fishsense_data/REEF/data \\
        --output rollover_scan.json

Common variants:

  - Limit to specific dives:
        --dive-id 237 --dive-id 249
  - Scan only canonical dives (default scans every dive):
        --canonical-only
  - Cap rollover-sibling depth (default 9 — supports up to PA999999):
        --max-rollover-depth 5
  - Tighten parallelism if the NAS pushes back (default 8 per dive):
        --concurrency 4
  - Only flag dives that look like rollover (contiguous tail of 404s):
        --require-contiguous-tail

Auth notes
----------
Per CLAUDE.md, `orchestrator.fishsense.e4e.ucsd.edu` is fronted by
authentik and SDK basic auth gets 302'd from a dev box. Either run
this from inside the orchestrator's docker network and point
`--api-url` at the internal address, or port-forward, or read images
out of Postgres directly and skip the API. The script will fail loudly
if the API returns HTML instead of JSON (httpx
`response.raise_for_status()` swallows the redirect chain only when
auth completes).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Iterable
from urllib.parse import urlparse

# Third-party deps installed by the api-workflow-worker package.
from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.image import Image
from synology_api.base_api import BaseApi
from synology_api.exceptions import FileStationError
from synology_api.filestation import FileStation


# Olympus filename: PA<5 digits>.ORF (case-insensitive). The frame-
# counter rollover happens at PA199999 -> PA000000 with the camera
# moving subsequent frames into a sibling directory. We don't filter
# on filename here — we check every image's path — but the regex is
# kept as documentation of what triggers Olympus's rollover behavior.
_OLYMPUS_FRAME_RE = re.compile(r"PA\d{5}\.ORF$", re.IGNORECASE)


@dataclass
class ImageScanResult:
    """Per-image outcome from the scan."""

    dive_id: int
    image_id: int | None
    image_path: str
    image_checksum: str
    status: str  # "ok", "rollover_resolved", "not_found", "no_path"
    suggested_rollover_path: str | None = None
    rollover_depth: int | None = None  # which sibling matched (1, 2, ...)


@dataclass
class DiveScanResult:  # pylint: disable=too-many-instance-attributes
    """Per-dive aggregate."""

    dive_id: int
    total_images: int
    ok: int = 0
    rollover_resolved: int = 0
    not_found: int = 0
    no_path: int = 0
    rollover_depths_seen: list[int] = field(default_factory=list)
    contiguous_tail_404: bool = False  # heuristic: is this the rollover symptom?


def _resolve_nas_path(nas_root: str, relative_path: str) -> str:
    """Mirror api-workflow-worker's `_resolve_nas_path` so the existence
    check uses the exact same prefix the staging activity would."""
    if relative_path.startswith("/"):
        return relative_path
    root = nas_root.rstrip("/")
    return f"{root}/{relative_path.lstrip('/')}"


def _rollover_candidates(image_path: str, max_depth: int) -> list[tuple[int, str]]:
    """Yield (depth, candidate_path) tuples for an image whose recorded
    path 404'd.

    Olympus convention: parent dive folder is named like
    `<prefix>_<rig>` (e.g. `101923_Alligator_FSL06`). On rollover the
    camera creates a child directory named `<prefix><N>_<rig>` for
    N=1, 2, ... and continues writing frames there. Rollovers can chain
    (PA999999 -> N=2, etc.).

    Returns candidates ordered by depth so the caller can take the
    first hit.
    """
    parts = image_path.rsplit("/", 1)
    if len(parts) != 2:
        return []
    parent_dir, filename = parts
    leaf = parent_dir.rsplit("/", 1)[-1]

    # Split `leaf` on the LAST underscore — that's `_<rig>`. Anything
    # before is the prefix we splice the digit into.
    if "_" not in leaf:
        return []
    prefix, rig = leaf.rsplit("_", 1)

    candidates: list[tuple[int, str]] = []
    for n in range(1, max_depth + 1):
        rollover_leaf = f"{prefix}{n}_{rig}"
        candidates.append((n, f"{parent_dir}/{rollover_leaf}/{filename}"))
    return candidates


def _is_contiguous_tail(per_image: list[ImageScanResult]) -> bool:
    """Return True iff every `not_found`/`rollover_resolved` row sits
    in a contiguous suffix of the (path-sorted) list. That's the
    signature of a frame-counter rollover: early frames in the dive
    are fine, late frames spill into the rollover dir.

    Sort by path so that filename ordering matches camera order (Olympus
    PA##### counters increment monotonically within a dive).
    """
    if not per_image:
        return False
    in_order = sorted(per_image, key=lambda r: r.image_path)
    seen_bad = False
    for row in in_order:
        bad = row.status in ("rollover_resolved", "not_found")
        if bad:
            seen_bad = True
        elif seen_bad:
            # A "good" row appeared after a "bad" one — not a tail.
            return False
    return seen_bad


class NasExistsClient:
    """Read-only wrapper around synology-api FileStation, exposing only
    `exists()`. Mirrors the api-worker's `NasClient.exists` so the scan
    behaves like the activity that consumes these paths."""

    def __init__(self, *, nas_url: str, username: str, password: str):
        parsed = urlparse(nas_url)
        if not parsed.hostname or not parsed.port:
            raise ValueError(
                f"NAS url must include hostname + port; got {nas_url!r}"
            )
        BaseApi.shared_session = None  # see api-worker NasClient note
        self._fs = FileStation(
            parsed.hostname,
            parsed.port,
            username,
            password,
            secure=True,
            cert_verify=False,
        )
        # synology-api's `search_start` (and a handful of other
        # methods) return a chatty human string instead of the
        # structured `{"taskid": ...}` dict when this flag is on.
        # Force the structured form so the orphan-locator can
        # extract the task id reliably.
        self._fs.interactive_output = False

    def exists(self, file_path: str) -> bool:
        try:
            info = self._fs.get_file_info(path=file_path)
        except FileStationError:
            return False
        if not isinstance(info, dict):
            return False
        files = info.get("data", {}).get("files", [])
        return any(isinstance(f, dict) and f.get("name") for f in files)

    def walk_orfs(self, folder_path: str, *, page_size: int = 1000) -> list[dict]:
        """Recursively walk `folder_path` via `get_file_list` and return
        every file whose name ends with `.ORF` (case-insensitive).

        Used as the orphan-locator's primary traversal mechanism.
        We tried `search_orfs()` first (DSM's async search task) but
        it kept returning `total=0` regardless of pattern/extension
        — likely because synology-api doesn't quote those params in
        the way DSM expects. Walking via `get_file_list` uses the
        same code path the per-file `exists()` check uses, which we
        already know works on this NAS for this user.
        """
        out: list[dict] = []
        stack: list[str] = [folder_path]
        dirs_walked = 0
        while stack:
            d = stack.pop()
            dirs_walked += 1
            if dirs_walked % 25 == 0:
                print(
                    f"      walked {dirs_walked} dir(s); "
                    f"{len(out)} ORF(s) so far; queue={len(stack)}",
                    flush=True,
                )
            offset = 0
            while True:
                resp = self._fs.get_file_list(
                    folder_path=d,
                    offset=offset,
                    limit=page_size,
                    additional=["type"],
                )
                if not isinstance(resp, dict):
                    break
                data = resp.get("data", {}) or {}
                files = data.get("files", []) or []
                if not files:
                    break
                for fr in files:
                    name = fr.get("name") or ""
                    path = fr.get("path") or ""
                    is_dir = bool(fr.get("isdir"))
                    if is_dir:
                        if path:
                            stack.append(path)
                        continue
                    if name.upper().endswith(".ORF") and path:
                        out.append(fr)
                offset += len(files)
                # If we got less than a full page, we've drained
                # this dir. Avoids one wasted round-trip per dir.
                if len(files) < page_size:
                    break
        print(
            f"      walked {dirs_walked} dir(s) total; "
            f"{len(out)} ORF(s) found",
            flush=True,
        )
        return out

    def search_orfs(
        self,
        folder_path: str,
        *,
        poll_interval: float = 1.0,
        max_wait: int = 600,
        page_size: int = 1000,
    ) -> list[dict]:
        """Recursively search `folder_path` for `*.ORF` files.

        Returns the raw FileStation file records (dicts with `name`,
        `path`, etc.). Used to build a basename → fullpath index for
        locating orphan files whose recorded `image.path` 404s and
        whose Olympus rollover heuristic also misses.

        FileStation's search is async: start a task, poll for results,
        stop the task when done. We page through all results — Synology
        caps `limit` server-side somewhere around a few thousand.

        NOTE: in practice this returned total=0 for our NAS regardless
        of filter; `walk_orfs()` is used instead. Kept here for the
        case where the search API does work and you want async/parallel
        traversal at scale.
        """
        # synology-api's `search_start` doesn't quote the `pattern`
        # / `extension` params before posting them, but DSM's
        # SYNO.FileStation.Search API expects them quoted (it does
        # quote `folder_path` and `filetype` via explicit code paths
        # — pattern/extension fall through a generic loop and arrive
        # unquoted). Both `extension="ORF"` and `pattern="*.ORF"`
        # silently return total=0 in practice. Easiest robust
        # workaround: don't ask DSM to filter, take everything under
        # the folder, then filter for `*.ORF` client-side. The
        # search-result records are small and the folder scope is
        # bounded, so this is fine.
        start_resp = self._fs.search_start(
            folder_path=folder_path,
            recursive=True,
        )
        if isinstance(start_resp, str):
            raise RuntimeError(f"FileStation search_start failed: {start_resp}")
        task_id = start_resp.get("taskid")
        if not task_id:
            raise RuntimeError(
                f"FileStation search_start returned no taskid: {start_resp!r}"
            )

        all_files: list[dict] = []
        offset = 0
        deadline = time.time() + max_wait
        consecutive_empty_polls = 0
        poll_count = 0
        try:
            while True:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"search task {task_id} did not finish in {max_wait}s"
                    )
                page = self._fs.get_search_list(
                    task_id=task_id, offset=offset, limit=page_size
                )
                if not isinstance(page, dict):
                    raise RuntimeError(
                        f"FileStation get_search_list returned: {page!r}"
                    )
                data = page.get("data", {}) or {}
                finished = bool(data.get("finished", False))
                total = int(data.get("total", 0) or 0)
                files = data.get("files", []) or []
                all_files.extend(files)
                offset += len(files)
                poll_count += 1
                print(
                    f"      poll #{poll_count}: finished={finished} "
                    f"total={total} got={len(files)} "
                    f"all_files_so_far={len(all_files)}",
                    flush=True,
                )

                # Defensive: if total is reported and we've collected
                # everything, stop. Avoids an extra empty round-trip
                # when the search finishes mid-page.
                if total and len(all_files) >= total:
                    break
                # Original break: search done AND no more rows on this
                # page. Add a one-poll grace period to dodge a
                # finished=True/files=[] transient that can occur
                # before DSM materializes the result set.
                if finished and not files:
                    consecutive_empty_polls += 1
                    if consecutive_empty_polls >= 2:
                        break
                    time.sleep(poll_interval)
                    continue
                consecutive_empty_polls = 0
                if not finished:
                    time.sleep(poll_interval)
        finally:
            try:
                self._fs.stop_search_task(f'"{task_id}"')
            # pylint: disable-next=broad-exception-caught
            except Exception:  # noqa: BLE001
                # Cleanup best-effort; the task will be GC'd by DSM eventually.
                pass

        return all_files


async def _scan_image(
    *,
    nas: NasExistsClient,
    nas_root: str,
    dive_id: int,
    image: Image,
    max_rollover_depth: int,
    sem: asyncio.Semaphore,
) -> ImageScanResult:
    if not image.path:
        return ImageScanResult(
            dive_id=dive_id,
            image_id=image.id,
            image_path="",
            image_checksum=image.checksum or "",
            status="no_path",
        )

    recorded_full = _resolve_nas_path(nas_root, image.path)

    async with sem:
        if await asyncio.to_thread(nas.exists, recorded_full):
            return ImageScanResult(
                dive_id=dive_id,
                image_id=image.id,
                image_path=image.path,
                image_checksum=image.checksum,
                status="ok",
            )

        for depth, candidate_relative in _rollover_candidates(
            image.path, max_rollover_depth
        ):
            candidate_full = _resolve_nas_path(nas_root, candidate_relative)
            if await asyncio.to_thread(nas.exists, candidate_full):
                return ImageScanResult(
                    dive_id=dive_id,
                    image_id=image.id,
                    image_path=image.path,
                    image_checksum=image.checksum,
                    status="rollover_resolved",
                    suggested_rollover_path=candidate_relative,
                    rollover_depth=depth,
                )

    return ImageScanResult(
        dive_id=dive_id,
        image_id=image.id,
        image_path=image.path,
        image_checksum=image.checksum,
        status="not_found",
    )


async def _scan_dive(
    *,
    api: Client,
    nas: NasExistsClient,
    nas_root: str,
    dive_id: int,
    max_rollover_depth: int,
    concurrency: int,
) -> tuple[DiveScanResult, list[ImageScanResult]]:
    images = await api.images.get(dive_id=dive_id) or []
    if not images:
        return DiveScanResult(dive_id=dive_id, total_images=0), []

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _scan_image(
            nas=nas,
            nas_root=nas_root,
            dive_id=dive_id,
            image=img,
            max_rollover_depth=max_rollover_depth,
            sem=sem,
        )
        for img in images
    ]
    per_image = await asyncio.gather(*tasks)

    summary = DiveScanResult(dive_id=dive_id, total_images=len(per_image))
    for row in per_image:
        if row.status == "ok":
            summary.ok += 1
        elif row.status == "rollover_resolved":
            summary.rollover_resolved += 1
            if row.rollover_depth is not None:
                summary.rollover_depths_seen.append(row.rollover_depth)
        elif row.status == "not_found":
            summary.not_found += 1
        elif row.status == "no_path":
            summary.no_path += 1
    summary.rollover_depths_seen = sorted(set(summary.rollover_depths_seen))
    summary.contiguous_tail_404 = _is_contiguous_tail(per_image)
    return summary, per_image


def _orphan_search_root(image_path: str, depth: int) -> str:
    """Walk `depth` levels up from `image_path`'s leaf-dir parent and
    return the resulting subdir (relative, no leading slash).

    `depth=1` → search inside the leaf-dir's parent (siblings of the
    recorded leaf dir, e.g. sibling rigs from the same dive day).
    `depth=2` → grandparent (whole day across rigs).
    `depth=3` → great-grandparent (often whole trip).
    Empty string → search the configured `nas_root` itself.
    """
    parts = image_path.split("/")
    # last component is the filename — drop it so parts[-1] is the
    # leaf dir (the one the rollover heuristic already covered).
    parts = parts[:-1]
    if depth >= len(parts):
        return ""
    return "/".join(parts[: len(parts) - depth])


def _resolve_search_root(nas_root: str, relative: str) -> str:
    if not relative:
        return nas_root.rstrip("/")
    if relative.startswith("/"):
        return relative
    return f"{nas_root.rstrip('/')}/{relative.lstrip('/')}"


def _strip_nas_root(nas_root: str, full_path: str) -> str:
    """Convert a NAS-absolute path back to the DB-relative form so the
    output rows speak the same `image.path` shape as the input."""
    root = nas_root.rstrip("/") + "/"
    if full_path.startswith(root):
        return full_path[len(root):]
    return full_path


def _disambiguate_by_prefix(
    recorded_path: str, candidates: list[str]
) -> tuple[list[str], int]:
    """Pick candidates that share the longest path-component prefix with
    `recorded_path`. Returns `(best, prefix_len)` where `best` is all
    candidates tied at the max prefix length.

    Olympus basenames recur across rigs (PA######.ORF) so a basename
    search across a whole trip yields N hits per file — one in each
    sibling rig dir. The right answer is almost always the one that
    shares the most path components with the recorded `image.path`,
    because that means same rig + same dive day + same parent
    structure. Anything mis-filed into a child of the recorded leaf
    (the actual symptom we're chasing) still scores higher than
    siblings under different rigs.
    """
    if not candidates:
        return [], 0
    rec_parts = recorded_path.split("/")

    def shared_len(cand: str) -> int:
        cand_parts = cand.split("/")
        n = 0
        for a, b in zip(rec_parts, cand_parts):
            if a == b:
                n += 1
            else:
                break
        return n

    scored = [(shared_len(c), c) for c in candidates]
    max_len = max(s for s, _ in scored)
    best = [c for s, c in scored if s == max_len]
    return best, max_len


def _locate_orphans_from_scan(  # pylint: disable=too-many-statements,too-many-locals
    *,
    nas: NasExistsClient,
    nas_root: str,
    input_path: str,
    output_path: str,
    search_depth: int,
) -> int:
    """Second-pass orphan locator: read a previous rollover-scan JSON,
    filter to `not_found` rows, and use FileStation's recursive search
    to find each missing file by basename within an ancestor of its
    recorded path.

    Writes a new JSON with the same shape as the input, plus
    `actually_found_at` populated for each row that was relocated.

    Why this is a separate pass instead of part of the rollover scan:
    rollover candidates are O(max_depth) HEAD calls per missing file
    — fast, scoped. Orphan search is one FileStation search task per
    distinct (search_root) that returns potentially thousands of
    files. We dedupe roots and run at most one search per root.
    """
    with open(input_path, encoding="utf-8") as f:
        prior = json.load(f)

    not_found_rows = [
        r for r in prior.get("affected_rows", []) if r.get("status") == "not_found"
    ]
    print(f"loaded {len(not_found_rows)} not_found rows from {input_path}")

    # Group rows by their search root (deduped) so we run one search
    # per root, regardless of how many missing files share it.
    roots_to_rows: dict[str, list[dict]] = {}
    for row in not_found_rows:
        root = _orphan_search_root(row["image_path"], search_depth)
        roots_to_rows.setdefault(root, []).append(row)

    print(
        f"running {len(roots_to_rows)} FileStation search(es) "
        f"(depth={search_depth} ancestors above leaf dir)"
    )

    # basename -> [list of full NAS paths] across all search roots.
    # If a basename appears in multiple search roots, we keep the
    # union — operator can disambiguate.
    location_index: dict[str, list[str]] = {}

    for root_relative in sorted(roots_to_rows.keys()):
        full_root = _resolve_search_root(nas_root, root_relative)
        row_count = len(roots_to_rows[root_relative])
        print(
            f"  searching {full_root}  "
            f"(covers {row_count} not_found row(s))",
            flush=True,
        )
        try:
            files = nas.walk_orfs(folder_path=full_root)
        # pylint: disable-next=broad-exception-caught
        except Exception as e:  # noqa: BLE001
            print(
                f"    [ERROR] walk failed: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(f"    -> {len(files)} ORF file(s) found", flush=True)
        for fr in files:
            name = fr.get("name")
            path = fr.get("path")
            if not name or not path:
                continue
            location_index.setdefault(name, []).append(path)

    # Match each not_found row's basename to the index, then
    # disambiguate the (usually multiple) hits by longest shared
    # path prefix with the recorded path.
    relocated = 0
    truly_missing = 0
    confident_matches = 0
    for row in not_found_rows:
        basename = row["image_path"].rsplit("/", 1)[-1]
        hits = location_index.get(basename, [])
        if hits:
            relative_hits = [_strip_nas_root(nas_root, p) for p in hits]
            row["actually_found_at"] = relative_hits
            row["actually_found_count"] = len(relative_hits)
            best, prefix_len = _disambiguate_by_prefix(
                row["image_path"], relative_hits
            )
            row["best_match_paths"] = best
            row["best_match_prefix_len"] = prefix_len
            row["confident_match"] = len(best) == 1
            if row["confident_match"]:
                confident_matches += 1
            relocated += 1
        else:
            row["actually_found_at"] = None
            row["actually_found_count"] = 0
            row["best_match_paths"] = None
            row["best_match_prefix_len"] = 0
            row["confident_match"] = False
            truly_missing += 1

    # Re-stitch the output: keep rollover_resolved rows untouched,
    # update not_found rows in place.
    by_id = {(r["dive_id"], r["image_id"]): r for r in not_found_rows}
    out_rows = []
    for r in prior.get("affected_rows", []):
        key = (r.get("dive_id"), r.get("image_id"))
        out_rows.append(by_id.get(key, r))

    output = dict(prior)
    output["affected_rows"] = out_rows
    output["orphan_locate_summary"] = {
        "search_depth": search_depth,
        "search_roots": sorted(roots_to_rows.keys()),
        "not_found_input_count": len(not_found_rows),
        "relocated_count": relocated,
        "truly_missing_count": truly_missing,
        "ambiguous_raw_count": sum(
            1 for r in not_found_rows if r.get("actually_found_count", 0) > 1
        ),
        "confident_after_disambiguation": confident_matches,
        "still_ambiguous_after_disambiguation": relocated - confident_matches,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print(f"relocated:                              {relocated}")
    print(f"truly missing:                          {truly_missing}")
    print(
        f"ambiguous (>1 raw hit before disambig): "
        f"{output['orphan_locate_summary']['ambiguous_raw_count']}"
    )
    print(f"confident after path-prefix disambig:   {confident_matches}")
    print(
        f"still ambiguous after disambig:         "
        f"{relocated - confident_matches}"
    )
    print(f"wrote -> {output_path}")
    return 0


async def _enumerate_dive_ids(
    api: Client, *, dive_ids: list[int], canonical_only: bool
) -> list[int]:
    if dive_ids:
        return sorted(set(dive_ids))
    if canonical_only:
        dives = await api.dives.get_canonical() or []
    else:
        dives = await api.dives.get() or []
    out = [d.id for d in dives if d.id is not None]
    return sorted(out)


def _print_progress(summary: DiveScanResult) -> None:
    flag = "ROLLOVER" if summary.rollover_resolved else (
        "MISSING " if summary.not_found else "ok      "
    )
    tail = " (contiguous tail)" if summary.contiguous_tail_404 else ""
    print(
        f"  [{flag}] dive_id={summary.dive_id:>6} "
        f"total={summary.total_images:>4} "
        f"ok={summary.ok:>4} "
        f"rollover={summary.rollover_resolved:>3} "
        f"missing={summary.not_found:>3} "
        f"no_path={summary.no_path:>3}"
        f"{tail}",
        flush=True,
    )


async def _run(args: argparse.Namespace) -> int:
    nas = NasExistsClient(
        nas_url=args.nas_url,
        username=args.nas_user,
        password=args.nas_pass,
    )

    async with Client(
        base_url=args.api_url,
        username=args.api_user,
        password=args.api_pass,
        timeout=args.api_timeout,
        max_concurrent_requests=args.api_concurrency,
    ) as api:
        dive_ids = await _enumerate_dive_ids(
            api,
            dive_ids=args.dive_id,
            canonical_only=args.canonical_only,
        )
        print(f"scanning {len(dive_ids)} dive(s)", flush=True)

        all_summaries: list[DiveScanResult] = []
        all_rows: list[ImageScanResult] = []

        for dive_id in dive_ids:
            try:
                summary, rows = await _scan_dive(
                    api=api,
                    nas=nas,
                    nas_root=args.nas_root,
                    dive_id=dive_id,
                    max_rollover_depth=args.max_rollover_depth,
                    concurrency=args.concurrency,
                )
            # pylint: disable-next=broad-exception-caught
            except Exception as e:  # noqa: BLE001
                # Log and keep going so a single bad dive doesn't
                # abort the whole scan.
                print(
                    f"  [ERROR  ] dive_id={dive_id} {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            _print_progress(summary)
            all_summaries.append(summary)
            all_rows.extend(rows)

    affected_summaries = [
        s for s in all_summaries
        if (s.rollover_resolved + s.not_found) > 0
        and (s.contiguous_tail_404 if args.require_contiguous_tail else True)
    ]
    affected_dive_ids = [s.dive_id for s in affected_summaries]

    print()
    print(f"scanned dives:          {len(all_summaries)}")
    print(f"affected dives:         {len(affected_summaries)}")
    print(f"  with rollover-resolved tail: "
          f"{sum(1 for s in affected_summaries if s.contiguous_tail_404)}")
    if affected_dive_ids:
        print(f"affected dive ids:      {affected_dive_ids}")
    print(f"rollover_resolved rows: "
          f"{sum(s.rollover_resolved for s in all_summaries)}")
    print(f"unresolved 404 rows:    "
          f"{sum(s.not_found for s in all_summaries)}")

    affected_rows = [
        r for r in all_rows
        if r.status in ("rollover_resolved", "not_found")
        and (
            (r.dive_id in set(affected_dive_ids))
            if args.require_contiguous_tail else True
        )
    ]

    output = {
        "scanned_dive_ids": [s.dive_id for s in all_summaries],
        "affected_dive_ids": affected_dive_ids,
        "summary_per_dive": [asdict(s) for s in all_summaries],
        "affected_rows": [asdict(r) for r in affected_rows],
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nwrote {len(affected_rows)} affected rows -> {args.output}")
    else:
        print()
        json.dump(output, sys.stdout, indent=2, default=str)
        print()

    return 0


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan fishsense-api Image rows for paths that 404 on the "
            "NAS, with Olympus rollover-sibling fallback."
        )
    )
    # API args are only required for the rollover scan. In
    # --locate-orphans-from mode the script reads dive/image data from
    # the prior scan JSON and never touches fishsense-api, so these
    # are optional.
    p.add_argument("--api-url")
    p.add_argument("--api-user")
    p.add_argument("--api-pass")
    p.add_argument("--api-timeout", type=int, default=30)
    p.add_argument("--api-concurrency", type=int, default=10)

    p.add_argument("--nas-url", required=True)
    p.add_argument("--nas-user", required=True)
    p.add_argument("--nas-pass", required=True)
    p.add_argument(
        "--nas-root",
        default="/fishsense_data/REEF/data",
        help="NAS prefix prepended to image.path (matches "
             "e4e_nas.raw_root_path on the worker; default: "
             "/fishsense_data/REEF/data).",
    )

    p.add_argument(
        "--dive-id",
        type=int,
        action="append",
        default=[],
        help="Restrict scan to specific dive id(s). Repeatable. "
             "Default: every dive.",
    )
    p.add_argument(
        "--canonical-only",
        action="store_true",
        help="Scan only canonical dives. Ignored if --dive-id is set.",
    )
    p.add_argument(
        "--max-rollover-depth",
        type=int,
        default=9,
        help="How many Olympus rollover siblings to try per 404. "
             "Default 9 (covers a single dive crossing PA199999 once "
             "with plenty of headroom).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel NAS exists() calls per dive. Default 8 (matches "
             "STAGE_CONCURRENCY in the staging activity).",
    )
    p.add_argument(
        "--require-contiguous-tail",
        action="store_true",
        help="Only count a dive as 'affected' if its 404s sit in a "
             "contiguous tail of the path-sorted image list — the "
             "signature of a frame-counter rollover. Use this to "
             "filter out random missing files when scanning broadly.",
    )
    p.add_argument(
        "--output",
        help="Write the JSON report to this file instead of stdout.",
    )

    # Orphan-locator pass: skip the rollover scan entirely and
    # FileStation-search for not_found basenames in the prior scan.
    p.add_argument(
        "--locate-orphans-from",
        metavar="SCAN_JSON",
        help="Skip the rollover scan and instead read a prior scan's "
             "JSON, then use FileStation recursive search to locate "
             "every `not_found` row by basename within an ancestor of "
             "its recorded path. Writes augmented rows to --output. "
             "Requires --output. API auth args are not used in this "
             "mode.",
    )
    p.add_argument(
        "--orphan-search-depth",
        type=int,
        default=3,
        help="In --locate-orphans-from mode, how many ancestors to "
             "walk above the recorded leaf dir before searching. "
             "1=siblings of leaf dir, 2=whole-day across rigs, 3=trip "
             "(default), 4+=trip+. Larger = more files scanned per "
             "search task. Empty path = whole nas-root.",
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.locate_orphans_from:
        if not args.output:
            print(
                "--locate-orphans-from requires --output", file=sys.stderr
            )
            return 2
        nas = NasExistsClient(
            nas_url=args.nas_url,
            username=args.nas_user,
            password=args.nas_pass,
        )
        return _locate_orphans_from_scan(
            nas=nas,
            nas_root=args.nas_root,
            input_path=args.locate_orphans_from,
            output_path=args.output,
            search_depth=args.orphan_search_depth,
        )
    # Rollover-scan mode requires API args.
    missing = [
        n for n, v in (
            ("--api-url", args.api_url),
            ("--api-user", args.api_user),
            ("--api-pass", args.api_pass),
        ) if not v
    ]
    if missing:
        print(
            f"missing required arg(s) for rollover scan: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

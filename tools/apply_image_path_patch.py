"""Apply the patch list produced by `scan_image_path_rollover.py`
(via `--locate-orphans-from`) to the prod fishsense Postgres
`image` table.

Why direct Postgres
-------------------
fishsense-api has no PUT/PATCH endpoint for `image.path` — verified
during the scan-tool design pass (see project memory). The only way
to fix the metadata is to UPDATE the table directly. Per CLAUDE.md
this is a prod-only DB ("No staging / test environment"), so the
script defaults to a dry-run and requires both `--apply` *and* an
interactive `YES` confirmation to commit. `--yes` skips the prompt
for scripted use.

Per-row safety
--------------
Each UPDATE is gated on `id = %s AND checksum = %s`. If either
shifted since the scan was generated, the UPDATE affects zero rows
and the script logs a `state-changed` warning rather than a
silently-mistargeted write. We also check the current `path` value:
if a row already carries the corrected path, we SKIP rather than
UPDATE — makes re-runs idempotent.

Source-of-correction
--------------------
For each row in the patch JSON:
  * `status == "rollover_resolved"` -> use `suggested_rollover_path`.
  * `status == "not_found"` AND `confident_match == True`
    -> use `best_match_paths[0]`.
Anything else is skipped (the script does not write paths it isn't
confident in).

Run
---
Dry-run (default — no writes):

    uv run --package fishsense-api python tools/apply_image_path_patch.py \\
        --patch-file rollover_scan_with_orphans.json \\
        --db-host fabricant-prod.ucsd.edu \\
        --db-name fishsense --db-user postgres --db-pass "$PG_PASS"

Apply for real:

    uv run --package fishsense-api python tools/apply_image_path_patch.py \\
        --patch-file rollover_scan_with_orphans.json \\
        --db-host fabricant-prod.ucsd.edu \\
        --db-name fishsense --db-user postgres --db-pass "$PG_PASS" \\
        --apply

Per-cohort rollout (recommended): `--dive-id 237` first, eyeball the
output, then re-run for the next dive. Whole-set apply with no
filter is supported but only as a last step.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

# psycopg is provided by services/fishsense-api's `psycopg[binary]` — installed
# only when the operator runs this tool from inside that service's venv, not
# part of any top-level workspace requirement, so pylint can't see it from the
# tool-level venv. Disable rather than add psycopg to a workspace dep group.
import psycopg  # pylint: disable=import-error


@dataclass
class PatchRow:
    """Normalized patch row — one corrected path per source row."""

    dive_id: int
    image_id: int
    image_checksum: str
    recorded_path: str
    corrected_path: str
    source: str  # "rollover" or "orphan"


def _derive_patch_rows(scan_json: dict) -> tuple[list[PatchRow], list[dict]]:
    """Walk `scan_json["affected_rows"]` and produce one PatchRow per
    row that has a confident corrected path. Returns `(patch_rows,
    skipped_unconfident_rows)`."""
    out: list[PatchRow] = []
    skipped: list[dict] = []
    for r in scan_json.get("affected_rows", []):
        status = r.get("status")
        corrected: str | None = None
        source: str = ""
        if status == "rollover_resolved":
            corrected = r.get("suggested_rollover_path")
            source = "rollover"
        elif status == "not_found" and r.get("confident_match"):
            best = r.get("best_match_paths") or []
            if len(best) == 1:
                corrected = best[0]
                source = "orphan"

        if not corrected:
            skipped.append(r)
            continue

        out.append(
            PatchRow(
                dive_id=r["dive_id"],
                image_id=r["image_id"],
                image_checksum=r["image_checksum"],
                recorded_path=r["image_path"],
                corrected_path=corrected,
                source=source,
            )
        )
    return out, skipped


def _classify_against_db(
    cur: psycopg.Cursor, row: PatchRow
) -> tuple[str, str | None]:
    """Look the row up in `image` table and decide what to do.

    Returns `(action, current_path_or_none)` where `action` is one of:
      * "update"           — current path matches recorded_path, ready to fix
      * "already-patched"  — current path already equals corrected_path
      * "drift-path"       — current path differs from BOTH recorded and
                             corrected. Someone else changed it; skip.
      * "drift-checksum"   — id matched but checksum doesn't. Skip.
      * "missing"          — no row with that id. Skip.
    """
    cur.execute(
        "SELECT path, checksum FROM image WHERE id = %s",
        (row.image_id,),
    )
    rec = cur.fetchone()
    if rec is None:
        return ("missing", None)
    current_path, current_checksum = rec
    if current_checksum != row.image_checksum:
        return ("drift-checksum", current_path)
    if current_path == row.corrected_path:
        return ("already-patched", current_path)
    if current_path != row.recorded_path:
        return ("drift-path", current_path)
    return ("update", current_path)


def _print_summary(
    patch_rows: list[PatchRow],
    classifications: dict[int, tuple[str, str | None]],
    skipped: list[dict],
) -> None:
    counts = Counter(action for action, _ in classifications.values())
    by_dive: dict[int, Counter] = {}
    for row in patch_rows:
        action = classifications[row.image_id][0]
        by_dive.setdefault(row.dive_id, Counter())[action] += 1

    print()
    print("=== overall ===")
    print(f"  patch_rows total:            {len(patch_rows)}")
    print(f"  unconfident rows skipped:    {len(skipped)}")
    print(f"  -> ready to UPDATE:          {counts.get('update', 0)}")
    print(f"  -> already patched (no-op):  {counts.get('already-patched', 0)}")
    print(f"  -> path drift (skip):        {counts.get('drift-path', 0)}")
    print(f"  -> checksum drift (skip):    {counts.get('drift-checksum', 0)}")
    print(f"  -> row missing (skip):       {counts.get('missing', 0)}")
    print()
    print("=== per dive ===")
    for did in sorted(by_dive):
        c = by_dive[did]
        print(
            f"  dive {did:>5}: "
            f"update={c.get('update', 0):>4} "
            f"already={c.get('already-patched', 0):>4} "
            f"drift-path={c.get('drift-path', 0):>3} "
            f"drift-checksum={c.get('drift-checksum', 0):>3} "
            f"missing={c.get('missing', 0):>3}"
        )


def _print_drift_examples(
    patch_rows: list[PatchRow],
    classifications: dict[int, tuple[str, str | None]],
    limit: int = 5,
) -> None:
    """Show a few examples of any drift state so the operator can
    eyeball whether the skip is the right call."""
    drift_rows = [
        (r, classifications[r.image_id])
        for r in patch_rows
        if classifications[r.image_id][0]
        in ("drift-path", "drift-checksum", "missing")
    ]
    if not drift_rows:
        return
    print()
    print(f"=== drift / missing examples (first {limit}) ===")
    for r, (action, current_path) in drift_rows[:limit]:
        print(f"  [{action}] dive={r.dive_id} image_id={r.image_id}")
        print(f"    recorded:   {r.recorded_path}")
        print(f"    db.path:    {current_path}")
        print(f"    corrected:  {r.corrected_path}")


def _execute_updates(
    cur: psycopg.Cursor,
    patch_rows: list[PatchRow],
    classifications: dict[int, tuple[str, str | None]],
) -> int:
    """Run an UPDATE for every row classified as 'update'. Returns the
    rowcount sum. The caller owns the transaction commit/rollback."""
    updated = 0
    for row in patch_rows:
        if classifications[row.image_id][0] != "update":
            continue
        cur.execute(
            "UPDATE image SET path = %s WHERE id = %s AND checksum = %s",
            (row.corrected_path, row.image_id, row.image_checksum),
        )
        if cur.rowcount != 1:
            # Should not happen — _classify_against_db already
            # confirmed the row matches. Bail loudly so the operator
            # can rollback the transaction.
            raise RuntimeError(
                f"UPDATE for image_id={row.image_id} affected "
                f"{cur.rowcount} rows, expected 1. "
                "Aborting so the transaction rolls back."
            )
        updated += 1
    return updated


def _confirm(updates_pending: int) -> bool:
    """Block on stdin for an explicit YES. Returns True if confirmed."""
    print()
    prompt = (
        f"About to COMMIT {updates_pending} UPDATE(s) against the prod "
        f"fishsense.image table. Type YES (uppercase) to commit, "
        "anything else to rollback: "
    )
    try:
        answer = input(prompt).strip()
    except EOFError:
        return False
    return answer == "YES"


def _run(args: argparse.Namespace) -> int:
    with open(args.patch_file, encoding="utf-8") as f:
        scan = json.load(f)

    patch_rows, skipped = _derive_patch_rows(scan)
    if args.dive_id:
        wanted = set(args.dive_id)
        patch_rows = [r for r in patch_rows if r.dive_id in wanted]
        print(f"filtered to dives {sorted(wanted)} -> {len(patch_rows)} row(s)")

    if not patch_rows:
        print("no patch rows to process — exiting.")
        return 0

    print(
        f"connecting to {args.db_user}@{args.db_host}:{args.db_port}/"
        f"{args.db_name}"
    )
    conninfo = (
        f"host={args.db_host} port={args.db_port} "
        f"dbname={args.db_name} user={args.db_user} "
        f"password={args.db_pass}"
    )

    # Single transaction: either every row commits or none. psycopg
    # opens an implicit transaction on first execute().
    with psycopg.connect(conninfo, autocommit=False) as conn:
        with conn.cursor() as cur:
            classifications: dict[int, tuple[str, str | None]] = {}
            for row in patch_rows:
                classifications[row.image_id] = _classify_against_db(cur, row)

            _print_summary(patch_rows, classifications, skipped)
            _print_drift_examples(patch_rows, classifications)

            updates_pending = sum(
                1 for a, _ in classifications.values() if a == "update"
            )

            if not args.apply:
                print()
                print(
                    f"DRY RUN — no writes executed. "
                    f"{updates_pending} UPDATE(s) would be applied. "
                    "Re-run with --apply to commit."
                )
                conn.rollback()  # be explicit even though no writes happened
                return 0

            if updates_pending == 0:
                print()
                print("Nothing to UPDATE — exiting without write.")
                conn.rollback()
                return 0

            if not args.yes and not _confirm(updates_pending):
                print()
                print("Confirmation declined — rolling back.")
                conn.rollback()
                return 1

            updated = _execute_updates(cur, patch_rows, classifications)
            conn.commit()
            print()
            print(f"COMMITTED {updated} UPDATE(s).")
            return 0


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply image.path patch rows from a scan JSON."
    )
    p.add_argument("--patch-file", required=True)
    p.add_argument("--db-host", required=True)
    p.add_argument("--db-port", type=int, default=5432)
    p.add_argument("--db-name", required=True)
    p.add_argument("--db-user", required=True)
    p.add_argument("--db-pass", required=True)
    p.add_argument(
        "--dive-id",
        type=int,
        action="append",
        default=[],
        help="Restrict to specific dive id(s). Repeatable. "
             "Default: every dive in the patch file.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Execute UPDATEs (within an interactive YES confirmation). "
             "Default is dry-run.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive YES prompt. Only meaningful with "
             "--apply. Use with care — bypasses the prod write gate.",
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())

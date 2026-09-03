"""Read-only detector for angle-sweep *runs* in a test dive's frames.

Background
----------
The model-yaw test dives were shot as a staged sweep: park the model at
a setpoint, take ~20 frames, rotate to the next setpoint, repeat. The
angle is therefore piecewise-constant over contiguous *runs* of frames,
and the labeling problem is not hundreds of per-frame judgements — it is
finding the boundaries and checking them.

The sweep is **nested**: a dive holds several blocks, one per (fish
model, distance) combination, and the angle sweep restarts inside each
block. So the structure is

    dive
      block   (one model at one distance)
        run   (one angle setpoint, ~20 frames)

which means angles are *cyclic*, not monotonic — 0,5..45, 0,5..45, ...
Swapping a model or moving the rig takes longer than rotating 5°, so the
two levels should separate on gap size alone: ~2s within a run, ~30s
between setpoints, minutes between blocks. `--report` shows both levels.

Neither of the other two variables needs labeling here. Distance is
*measured*, not annotated — it is `LaserDepth.depth_m`, already computed
per image. Model identity comes from `Fish.name` on the existing species
labeling path, which is what `fish_model_measurement_accuracy` already
joins through. Angle is the only genuinely new fact.

Rotating the model takes longer than the interval between shots, so
each boundary should show up as a gap in `Image.taken_datetime` (exact
per-frame EXIF 0x0132, already in the DB). This script measures that,
proposes a split, and emits a run table for a human to verify.

The intended chain is **generate -> human verifies -> load**:

    1. `--report`  does the timestamp gap actually separate the runs?
    2. `--split`   emit a run table with angles pre-filled monotonically.
    3. (human edits / confirms the CSV)
    4. `--expand`  turn the verified run table into `checksum,angle_deg`.

Step 4's output is what a loader PUTs against the API. This script
never writes anywhere: it issues SELECTs and writes local CSVs.

Does the split actually work?
-----------------------------
`--report` answers that with the **separation ratio** — the smallest
between-run gap divided by the largest within-run gap at the chosen
threshold. Comfortably above ~5 means the boundaries are unambiguous
and the auto split can be trusted after a spot check. Near 1 means the
shooting interval and the rotation pause overlap, the timestamps cannot
tell them apart, and you need eyes on frames instead (a numbered
contact sheet in timestamp order beats a Label Studio project for
this).

A wrong boundary silently mislabels ~20 frames, so two guardrails are
worth using every time:

  * `--expected-runs N` fails the dive rather than emitting a table
    whose run count disagrees with the number of setpoints you shot.
  * The run-size column is the tell — "take 20 at each angle" should
    give runs of ~20. A run of 3, or of 137, is a merged or split
    boundary, not a real setpoint.

Monotonic angle pre-fill assumes the sweep was shot in setpoint order.
Spot-check the first and last run; that catches a reversed or restarted
sweep, which is the failure the run sizes cannot show you.

Credentials
-----------
Reads the DSN from `$FISHSENSE_DSN` — never from a config file, so no
secret is read into a transcript:

    export FISHSENSE_DSN='postgresql://user:pass@host:5432/fishsense'

Run
---
    uv run python tools/detect_angle_sweep_runs.py --dive-id 512 --report

    uv run python tools/detect_angle_sweep_runs.py --dive-id 512 \\
        --split --gap-seconds 30 --block-gap-seconds 180 \\
        --setpoints-per-block 10 --start-deg 0 --step-deg 5 \\
        --output dive512_runs.csv

    uv run python tools/detect_angle_sweep_runs.py \\
        --expand dive512_runs.csv --output dive512_angles.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg

# Timestamps come from EXIF at one-second resolution, so a raw ratio
# against a zero gap (two frames in the same second, common in burst
# mode) is meaningless. Regularize the denominator by half a second.
_GAP_EPS_S = 0.5

# Only consider a split into at most this many runs when auto-picking a
# threshold. A sweep has tens of setpoints, not hundreds; without the
# cap the largest ratio is often found deep in the noise floor.
_MAX_AUTO_RUNS = 200

# A jump in the gap spectrum has to be at least this much bigger than the
# level below it to count as structure rather than jitter. Measured on
# synthetic sweeps: real setpoint boundaries clear 8x comfortably, while a
# shooting interval that overlaps the rotation pause sits near 1.3x.
_MIN_STRUCTURE_RATIO = 3.0


@dataclass(frozen=True)
class Frame:
    """One canonical frame of a dive, in capture order."""

    image_id: int
    checksum: str
    path: str
    taken_datetime: datetime

    @property
    def basename(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Run:
    """A contiguous group of frames believed to share one setpoint."""

    index: int
    frames: tuple[Frame, ...]


def fetch_frames(
    conn: psycopg.Connection, dive_id: int, include_noncanonical: bool
) -> list[Frame]:
    """Return the dive's frames ordered by capture time.

    Non-canonical rows are excluded by default: the same physical frame
    legitimately appears under several dive rows, and a duplicate would
    show up here as a zero gap that dilutes the separation ratio.
    """
    sql = """
        SELECT id, checksum, path, taken_datetime
        FROM image
        WHERE dive_id = %s
          AND taken_datetime IS NOT NULL
    """
    if not include_noncanonical:
        sql += " AND is_canonical"
    sql += " ORDER BY taken_datetime, id"

    with conn.cursor() as cur:
        cur.execute(sql, (dive_id,))
        return [Frame(*row) for row in cur.fetchall()]


def dive_label(conn: psycopg.Connection, dive_id: int) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT name, path FROM dive WHERE id = %s", (dive_id,))
        row = cur.fetchone()
    if row is None:
        return f"#{dive_id} (no such dive)"
    name, path = row
    return f"{name or path} #{dive_id}"


def resolve_dive_ids(
    conn: psycopg.Connection, dive_ids: list[int], name_like: str | None
) -> list[int]:
    resolved = list(dive_ids)
    if name_like:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM dive WHERE name ILIKE %s OR path ILIKE %s"
                " ORDER BY id",
                (name_like, name_like),
            )
            resolved.extend(r[0] for r in cur.fetchall())
    return sorted(set(resolved))


def gaps_seconds(frames: list[Frame]) -> list[float]:
    """Inter-frame gaps, so `gaps[i]` precedes `frames[i + 1]`."""
    return [
        (frames[i + 1].taken_datetime - frames[i].taken_datetime).total_seconds()
        for i in range(len(frames) - 1)
    ]


def auto_gap_threshold(
    gaps: list[float], min_ratio: float = _MIN_STRUCTURE_RATIO
) -> float | None:
    """Propose a split threshold at the *lowest* strong jump in the spectrum.

    Lowest, not largest, and the distinction is the whole point on a
    nested sweep. Gaps here are trimodal — ~2s within a run, ~30s between
    setpoints, minutes between blocks — and with a handful of blocks the
    30s-to-300s jump can out-rank the 2s-to-30s one. Taking the largest
    jump then silently collapses every setpoint in a block into a single
    run, which is the worst available answer: it looks like a clean split
    and mislabels everything.

    Scanning up from the noise floor instead finds the finest real
    structure, which is what a *run* boundary is. Feeding the resulting
    boundary gaps back through this same function then finds the next
    level up, which is what a *block* boundary is.

    Returns None when no jump clears `min_ratio` — i.e. the gaps are one
    smooth continuum and timestamps cannot find structure in them. That
    is a real answer, not a failure, and better than naming a threshold
    off the noise floor.

    Still a *hint*: it has no idea how many setpoints you shot. Cross-check
    with `--setpoints-per-block` and the run sizes.
    """
    if len(gaps) < 2:
        return None
    ordered = sorted(gaps, reverse=True)
    limit = min(len(ordered) - 1, _MAX_AUTO_RUNS)
    for k in range(limit - 1, -1, -1):
        if ordered[k] / (ordered[k + 1] + _GAP_EPS_S) >= min_ratio:
            return ordered[k]
    return None


def separation_ratio(gaps: list[float], threshold: float) -> float | None:
    """min(between-run gap) / max(within-run gap) at `threshold`.

    The honest answer to "can timestamps alone find the boundaries?".
    None when the threshold puts everything on one side.
    """
    between = [g for g in gaps if g >= threshold]
    within = [g for g in gaps if g < threshold]
    if not between or not within:
        return None
    return min(between) / (max(within) + _GAP_EPS_S)


def split_runs(frames: list[Frame], threshold: float) -> list[Run]:
    if not frames:
        return []
    runs: list[list[Frame]] = [[frames[0]]]
    for previous, current in zip(frames, frames[1:]):
        delta = (current.taken_datetime - previous.taken_datetime).total_seconds()
        if delta >= threshold:
            runs.append([current])
        else:
            runs[-1].append(current)
    return [Run(i + 1, tuple(f)) for i, f in enumerate(runs)]


def boundary_gaps(runs: list[Run]) -> list[float]:
    """The gap that created each run boundary (one shorter than `runs`)."""
    return [
        (b.frames[0].taken_datetime - a.frames[-1].taken_datetime).total_seconds()
        for a, b in zip(runs, runs[1:])
    ]


def assign_blocks(runs: list[Run], threshold: float | None) -> list[int]:
    """Group runs into blocks, one per (model, distance) combination.

    A block boundary is a run boundary whose gap is itself large —
    swapping a fish model or moving the rig costs minutes where a 5°
    rotation costs seconds. With no threshold the whole dive is one
    block, which is the right answer for a single-model single-distance
    dive.
    """
    blocks = [1]
    if threshold is None:
        return blocks * len(runs)
    for gap in boundary_gaps(runs):
        blocks.append(blocks[-1] + 1 if gap >= threshold else blocks[-1])
    return blocks


def histogram(gaps: list[float]) -> list[tuple[str, int]]:
    edges = [0, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600]
    buckets: list[tuple[str, int]] = []
    for low, high in zip(edges, edges[1:]):
        n = sum(1 for g in gaps if low <= g < high)
        buckets.append((f"{low:>4}-{high:<4}s", n))
    buckets.append((f"{edges[-1]:>4}+    s", sum(1 for g in gaps if g >= edges[-1])))
    return buckets


def report(conn: psycopg.Connection, dive_id: int, args: argparse.Namespace) -> None:
    frames = fetch_frames(conn, dive_id, args.include_noncanonical)
    print(f"\n=== {dive_label(conn, dive_id)} ===")
    if len(frames) < 2:
        print(f"  {len(frames)} usable frame(s) — nothing to split.")
        return

    gaps = gaps_seconds(frames)
    span = frames[-1].taken_datetime - frames[0].taken_datetime
    print(f"  frames: {len(frames)}   span: {span}")
    print(f"  first:  {frames[0].basename} @ {frames[0].taken_datetime}")
    print(f"  last:   {frames[-1].basename} @ {frames[-1].taken_datetime}")

    print("\n  gap histogram:")
    for band, count in histogram(gaps):
        if count:
            print(f"    {band}  {'#' * min(count, 60)} {count}")

    ordered = sorted(gaps, reverse=True)
    print("\n  largest gaps: " + ", ".join(f"{g:.0f}s" for g in ordered[:25]))
    median_gap = sorted(gaps)[len(gaps) // 2]
    print(f"  typical gap:  {median_gap:.1f}s (median)")

    threshold = args.gap_seconds or auto_gap_threshold(gaps)
    source = "given" if args.gap_seconds else "auto"
    if threshold is None:
        weak = auto_gap_threshold(gaps, min_ratio=1.0)
        if weak is None:
            print("\n  no structure at all in the gaps.")
            return
        print(
            f"\n  NO CLEAN SPLIT: the biggest jump in the gap spectrum is at "
            f"{weak:.0f}s but is under {_MIN_STRUCTURE_RATIO:g}x the level "
            "below it. Timestamps cannot separate these runs — use a numbered "
            "contact sheet in capture order instead."
        )
        threshold, source = weak, "weak"
    runs = split_runs(frames, threshold)
    ratio = separation_ratio(gaps, threshold)
    sizes = [len(r.frames) for r in runs]

    print(f"\n  threshold ({source}): {threshold:.0f}s -> {len(runs)} runs")
    if ratio is None:
        print("  separation:  n/a (threshold puts every gap on one side)")
    else:
        verdict = (
            "clean — auto split is trustworthy after a spot check"
            if ratio >= 5
            else (
                "weak — check the boundaries by eye"
                if ratio >= 2
                else "NOT separable — timestamps cannot find these boundaries"
            )
        )
        print(f"  separation:  {ratio:.1f}x  ({verdict})")
    median_size = sorted(sizes)[len(sizes) // 2]
    print(f"  run sizes:   min {min(sizes)}  median {median_size}  max {max(sizes)}")
    # Half or double the median run is a merged or split boundary, not a
    # setpoint you actually shot.
    odd = [
        r.index
        for r in runs
        if not 0.5 * median_size <= len(r.frames) <= 2 * median_size
    ]
    if odd:
        print(f"  ODD RUNS:    {odd} — likely a merged or split boundary")

    # Second level: which run boundaries are model/distance changes?
    edges = boundary_gaps(runs)
    block_threshold = args.block_gap_seconds
    if block_threshold is None:
        block_threshold = auto_gap_threshold(edges)

    blocks = assign_blocks(runs, block_threshold)
    n_blocks = blocks[-1]
    if block_threshold is None:
        print(
            "\n  blocks:      1 (no second-level gap found — single model "
            "and distance, or the swap was as quick as a rotation)"
        )
    else:
        per_block = [blocks.count(b) for b in range(1, n_blocks + 1)]
        source = "given" if args.block_gap_seconds else "auto"
        print(f"\n  block threshold ({source}): {block_threshold:.0f}s")
        print(f"  blocks:      {n_blocks}  runs per block: {per_block}")
        if len(set(per_block)) != 1:
            print(
                "  UNEVEN:      blocks disagree on setpoint count — either a "
                "block was cut short or a boundary was missed"
            )

    expected = args.setpoints_per_block
    if expected:
        if len(runs) % expected:
            print(
                f"  MISMATCH:    {len(runs)} runs is not a multiple of "
                f"{expected} setpoints per block"
            )
        else:
            print(f"  implies:     {len(runs) // expected} blocks of {expected}")
        top = args.start_deg + args.step_deg * (expected - 1)
        print(f"  angle range: {args.start_deg:g}..{top:g}° in {args.step_deg:g}° steps")
    if args.expected_runs and len(runs) != args.expected_runs:
        print(
            f"  MISMATCH:    expected {args.expected_runs} runs, "
            f"found {len(runs)}"
        )


def write_run_table(
    conn: psycopg.Connection, dive_id: int, args: argparse.Namespace, out: Path
) -> int:
    frames = fetch_frames(conn, dive_id, args.include_noncanonical)
    if len(frames) < 2:
        print(f"dive {dive_id}: too few frames", file=sys.stderr)
        return 1

    gaps = gaps_seconds(frames)
    threshold = args.gap_seconds or auto_gap_threshold(gaps)
    if threshold is None:
        print(f"dive {dive_id}: no threshold", file=sys.stderr)
        return 1

    runs = split_runs(frames, threshold)
    if args.expected_runs and len(runs) != args.expected_runs:
        print(
            f"dive {dive_id}: expected {args.expected_runs} runs, found "
            f"{len(runs)} at a {threshold:.0f}s threshold — refusing to write "
            "a run table. Re-run --report and pick a threshold by hand.",
            file=sys.stderr,
        )
        return 2

    block_threshold = args.block_gap_seconds
    if block_threshold is None:
        block_threshold = auto_gap_threshold(boundary_gaps(runs))
    blocks = assign_blocks(runs, block_threshold)
    per_block = [blocks.count(b) for b in range(1, blocks[-1] + 1)]

    # The nesting guardrail. A dive with several models or distances holds
    # one full angle sweep per block, so an uneven block is a boundary this
    # script got wrong -- and getting it wrong shifts every angle after it.
    expected = args.setpoints_per_block
    if expected and set(per_block) != {expected}:
        print(
            f"dive {dive_id}: blocks hold {per_block} runs, expected "
            f"{expected} setpoints each — refusing to write a run table. "
            "Adjust --block-gap-seconds, or fill the angles by hand.",
            file=sys.stderr,
        )
        return 2

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "dive_id",
                "block",
                "run",
                "run_in_block",
                "first",
                "last",
                "n",
                "angle_deg",
                "first_image_id",
            ]
        )
        seen: dict[int, int] = {}
        for run, block in zip(runs, blocks):
            seen[block] = seen.get(block, 0) + 1
            # Cyclic, not monotonic: the sweep restarts in each block.
            angle = args.start_deg + args.step_deg * (seen[block] - 1)
            writer.writerow(
                [
                    dive_id,
                    block,
                    run.index,
                    seen[block],
                    run.frames[0].basename,
                    run.frames[-1].basename,
                    len(run.frames),
                    angle,
                    run.frames[0].image_id,
                ]
            )
    top = args.start_deg + args.step_deg * (max(per_block) - 1)
    print(
        f"dive {dive_id}: wrote {len(runs)} runs in {blocks[-1]} block(s) to "
        f"{out} (run threshold {threshold:.0f}s, block threshold "
        f"{block_threshold:.0f}s, angles {args.start_deg:g}..{top:g}° step "
        f"{args.step_deg:g} restarting each block) — VERIFY before expanding. "
        "Check the FIRST and LAST run of each block: a sweep shot in reverse "
        "gets exactly the wrong angles and nothing else here will notice."
    )
    return 0


def expand(conn: psycopg.Connection, table: Path, out: Path) -> int:
    """Turn a verified run table into per-frame `checksum,angle_deg` rows."""
    with table.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    blanks = [r["run"] for r in rows if not (r.get("angle_deg") or "").strip()]
    if blanks:
        print(
            f"{table}: runs {blanks} have no angle_deg — fill them in first.",
            file=sys.stderr,
        )
        return 2

    written = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["image_id", "checksum", "path", "angle_deg", "dive_id", "block"]
        )
        for dive_id in sorted({int(r["dive_id"]) for r in rows}):
            frames = fetch_frames(conn, dive_id, include_noncanonical=False)
            by_name = {f.basename: i for i, f in enumerate(frames)}
            for row in (r for r in rows if int(r["dive_id"]) == dive_id):
                start, end = by_name.get(row["first"]), by_name.get(row["last"])
                if start is None or end is None:
                    print(
                        f"dive {dive_id} run {row['run']}: "
                        f"{row['first']}..{row['last']} not found in the dive.",
                        file=sys.stderr,
                    )
                    return 2
                for frame in frames[start : end + 1]:
                    writer.writerow(
                        [
                            frame.image_id,
                            frame.checksum,
                            frame.path,
                            row["angle_deg"],
                            dive_id,
                            row.get("block", 1),
                        ]
                    )
                    written += 1
    print(f"wrote {written} per-frame rows to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dive-id", type=int, action="append", default=[])
    parser.add_argument(
        "--dive-name-like", help="SQL ILIKE pattern matched against name and path"
    )
    parser.add_argument("--report", action="store_true", help="gap analysis only")
    parser.add_argument("--split", action="store_true", help="write a run table")
    parser.add_argument("--expand", type=Path, help="verified run table to expand")
    parser.add_argument(
        "--gap-seconds",
        type=float,
        help="split threshold; omitted means auto-pick and report the choice",
    )
    parser.add_argument(
        "--block-gap-seconds",
        type=float,
        help="second-level threshold separating (model, distance) blocks; "
        "omitted means auto-detect and report the choice",
    )
    parser.add_argument(
        "--setpoints-per-block",
        type=int,
        help="angles per block (0..45 in 5° steps is 10) — refuses to write "
        "a run table unless every block holds exactly this many runs",
    )
    parser.add_argument("--expected-runs", type=int, help="total run count guardrail")
    parser.add_argument("--start-deg", type=float, default=0.0)
    parser.add_argument("--step-deg", type=float, default=5.0)
    parser.add_argument("--include-noncanonical", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dsn = os.environ.get("FISHSENSE_DSN")
    if not dsn:
        print(
            "Set FISHSENSE_DSN, e.g.\n"
            "  export FISHSENSE_DSN='postgresql://user:pass@host:5432/fishsense'",
            file=sys.stderr,
        )
        return 2

    with psycopg.connect(dsn) as conn:
        if args.expand:
            if not args.output:
                print("--expand needs --output", file=sys.stderr)
                return 2
            return expand(conn, args.expand, args.output)

        dive_ids = resolve_dive_ids(conn, args.dive_id, args.dive_name_like)
        if not dive_ids:
            print(
                "No dives matched. Use --dive-id or --dive-name-like.",
                file=sys.stderr,
            )
            return 2

        if args.split:
            if not args.output:
                print("--split needs --output", file=sys.stderr)
                return 2
            if len(dive_ids) > 1:
                print("--split takes one dive at a time", file=sys.stderr)
                return 2
            return write_run_table(conn, dive_ids[0], args, args.output)

        for dive_id in dive_ids:
            report(conn, dive_id, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

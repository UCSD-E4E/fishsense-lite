"""Audit a dive's laser calibration by the range trend of its rigid objects.

A rigid object must read the same length at every range. An in-plane error in
the laser calibration -- the one term reprojection residual, the baseline gate
and the dot-on-board check are all blind to -- makes it read a length that
changes linearly with range, at (error / baseline) per metre. This script
fits that slope per (dive, object) over the dive's own `Measurement` and
`LaserDepth` rows and reports the implied angle. No known length is used, so
the check is not circular; see `range_trend.py` for what it measured on the
corpus and for why only the negative side flags.

It reads `Measurement`, `LaserDepth`, `SpeciesLabel` and `LaserExtrinsics`
through the SDK and writes nothing. Run it after stage 14 and the depth stage
have visited the dive; a dive with fewer than 8 frames of one object beyond
0.8 m, or a range spread under 2x, reports "insufficient" rather than a
number.

USAGE
    uv run --package fishsense-data-processing-workflow-worker python \\
        services/fishsense-data-processing-workflow-worker/scripts/\\
audit_length_range_trend.py 490 491 492
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import numpy as np
from fishsense_api_sdk.client import Client
from fishsense_shared.taxonomy import parse_model_name

from fishsense_data_processing_workflow_worker.config import settings
from fishsense_data_processing_workflow_worker.range_trend import (
    DEFAULT_MIN_DEPTH_M,
    DEFAULT_MIN_FRAMES,
    DEFAULT_MIN_RANGE_RATIO,
    RangeTrend,
    group_by_object,
    range_trend,
)


async def audit_dive(
    fs: Client, dive_id: int, min_frames: int, min_range_ratio: float
) -> dict[str, RangeTrend | None]:
    """Range trend per rigid object on `dive_id`; None where the data cannot
    support a slope."""
    extrinsics = await fs.dives.get_laser_extrinsics(dive_id)
    if extrinsics is None or extrinsics.laser_position is None:
        print(f"dive {dive_id}: no resolvable laser extrinsics", file=sys.stderr)
        return {}
    baseline_m = float(np.hypot(*np.asarray(extrinsics.laser_position)[:2]))

    measurements = await fs.fish.get_measurements(dive_id) or []
    depths = {
        d.image_id: float(d.depth_m)
        for d in await fs.images.get_laser_depths(dive_id)
        if d.depth_m is not None and d.depth_m > 0
    }
    names = {
        s.image_id: parse_model_name(s.content_of_image)
        for s in (await fs.labels.get_species_labels(dive_id) or [])
        if s.completed and not s.superseded
    }
    groups = group_by_object(measurements, depths, names)
    return {
        name: range_trend(
            zs,
            ls,
            baseline_m,
            min_frames=min_frames,
            min_range_ratio=min_range_ratio,
        )
        for name, (zs, ls) in sorted(groups.items())
    }


def report(dive_id: int, trends: dict[str, RangeTrend | None]) -> None:
    """One line per object; the note carries the interpretation."""
    print(f"\n=== dive {dive_id} ===")
    if not trends:
        print("  no measured rigid objects")
        return
    print(
        f"  {'object':<16} {'n':>3} {'range (m)':>11} {'slope %/m':>10} "
        f"{'95% CI':>17} {'angle':>8}  note"
    )
    for name, t in trends.items():
        if t is None:
            print(
                f"  {name:<16} insufficient (need >= {DEFAULT_MIN_FRAMES} frames "
                f"beyond {DEFAULT_MIN_DEPTH_M} m spanning >= "
                f"{DEFAULT_MIN_RANGE_RATIO}x)"
            )
            continue
        flag = "FLAG " if t.flagged else "     "
        zlo, zhi = t.depth_range_m
        lo, hi = t.ci_pct_per_m
        print(
            f"  {name:<16} {t.n:>3} {zlo:4.2f}-{zhi:4.2f} "
            f"{t.slope_pct_per_m:>+10.2f} [{lo:+6.2f},{hi:+6.2f}] "
            f"{t.eps_deg:>+7.3f}d  {flag}{t.note}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dive_ids", type=int, nargs="+")
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument(
        "--min-range-ratio", type=float, default=DEFAULT_MIN_RANGE_RATIO
    )
    args = parser.parse_args()

    async with Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    ) as fs:
        for dive_id in args.dive_ids:
            report(
                dive_id,
                await audit_dive(fs, dive_id, args.min_frames, args.min_range_ratio),
            )


if __name__ == "__main__":
    asyncio.run(main())

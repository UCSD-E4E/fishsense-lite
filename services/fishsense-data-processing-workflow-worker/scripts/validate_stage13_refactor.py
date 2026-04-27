"""Phase-1 read-only validation for the stage13 laser-calibration refactor.

The refactor delegates the projected-ray and atanasov-calibration math to
fishsense_core (WorldPointHandler.project_image_point + laser.calibrate_laser).
Stage13 has not been re-run since that refactor landed.

This script samples dives that ALREADY have laser_extrinsics computed by the
deployed pre-refactor implementation in production, recomputes extrinsics
with the refactored pipeline (same math as stage13), and reports numerical
deltas. It NEVER calls put_laser_extrinsics — pure read-only validation,
using the existing prod values as a free oracle.

Usage:
    uv run --directory services/fishsense-data-processing-workflow-worker \\
        python scripts/validate_stage13_refactor.py
"""

import asyncio
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fishsense_api_sdk.client import Client
from fishsense_core.laser import calibrate_laser as _calibrate_laser
from fishsense_core.world_point import WorldPointHandler

from fishsense_data_processing_workflow_worker.config import settings

INCH_TO_M = 0.0254
SAMPLE_SIZE = 30
RANDOM_SEED = 0xF15
AXIS_ANGLE_TOLERANCE_DEG = 0.5
POSITION_L2_TOLERANCE_M = 0.001


def image_coordinate_to_projected_point(
    image_point: np.ndarray, k_inv: np.ndarray
) -> np.ndarray:
    return WorldPointHandler(k_inv).project_image_point(image_point)


def get_normal_vector_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, 2]


def atanasov_calibration_method(ps: np.ndarray) -> np.ndarray:
    origin, orientation = _calibrate_laser(ps.astype(np.float32))
    return np.concatenate([orientation.astype(float), origin[:2].astype(float)])


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


async def compute_extrinsics(
    client: Client, dive, dive_slate, dive_slate_labels, camera_intrinsics
):
    """Stage13's per-dive pipeline. Returns (axis, position, n_points_used, skipped)."""
    laser_points: list[np.ndarray] = []
    skipped: list[tuple[str, int, str]] = []

    for label in dive_slate_labels:
        try:
            laser_label = await client.labels.get_laser_label(label.image_id)
        except Exception as e:  # pylint: disable=broad-except
            skipped.append(("get_laser_label", label.image_id, repr(e)))
            continue

        source_points = list(dive_slate.reference_points)
        for idx in label.skipped_points or []:
            source_points.pop(idx)
        source_points = np.array(source_points)

        body_points = np.zeros((len(source_points), 3), dtype=np.float32)
        body_points[:, :2] = (source_points / float(dive_slate.dpi)) * INCH_TO_M
        image_space = np.array(label.reference_points)

        ret, rvec, tvec = cv2.solvePnP(
            body_points,
            image_space,
            camera_intrinsics.camera_matrix,
            np.zeros((5,)),
        )
        if not ret:
            skipped.append(("solvePnP", label.image_id, "ret=False"))
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_space = (rotation @ body_points.T + tvec).T
        normal = get_normal_vector_from_rotation_matrix(rotation)

        laser_image_point = np.array([laser_label.x, laser_label.y])
        ray = (
            image_coordinate_to_projected_point(
                laser_image_point, np.linalg.inv(camera_intrinsics.camera_matrix)
            )
            * -1
        )
        if np.any(np.isnan(ray)):
            skipped.append(("ray-nan", label.image_id, ""))
            continue

        scale = (normal.T @ camera_space[0, :]) / (normal.T @ ray)
        laser_points.append(ray * scale)

    if len(laser_points) < 2:
        raise ValueError(f"insufficient laser points: {len(laser_points)}")

    params = atanasov_calibration_method(np.array(laser_points))
    new_axis = params[:3]
    new_position = np.zeros(3, dtype=float)
    new_position[:2] = params[-2:]
    return new_axis, new_position, len(laser_points), skipped


async def main() -> int:
    rng = random.Random(RANDOM_SEED)
    print(f"Connecting to {settings.fishsense_api.url}")

    async with Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    ) as client:
        dives = await client.dives.get()
        dives = [d for d in dives if d.priority == "HIGH"]
        print(f"HIGH-priority dives: {len(dives)}")

        extrinsics = await asyncio.gather(
            *[client.dives.get_laser_extrinsics(d.id) for d in dives]
        )
        with_calib = [(d, le) for d, le in zip(dives, extrinsics) if le]
        print(f"  ...with existing laser_extrinsics: {len(with_calib)}")
        if not with_calib:
            print("Nothing to compare against. Aborting.")
            return 1

        sample = rng.sample(with_calib, min(SAMPLE_SIZE, len(with_calib)))
        print(f"Sampled {len(sample)} dives (seed={RANDOM_SEED}).\n")

        dive_slates = await client.dive_slates.get()
        slates_by_id = {ds.id: ds for ds in dive_slates}

        results: list[dict] = []
        for i, (dive, stored) in enumerate(sample, 1):
            print(f"[{i:2d}/{len(sample)}] dive_id={dive.id} camera_id={dive.camera_id}")
            try:
                if dive.dive_slate_id is None or dive.dive_slate_id not in slates_by_id:
                    print(f"  SKIP: no resolvable dive_slate ({dive.dive_slate_id})")
                    results.append({"dive_id": dive.id, "status": "no_slate"})
                    continue
                slate = slates_by_id[dive.dive_slate_id]
                labels = await client.labels.get_dive_slate_labels(dive.id)
                if not labels:
                    print("  SKIP: no dive_slate_labels")
                    results.append({"dive_id": dive.id, "status": "no_labels"})
                    continue
                intrinsics = await client.cameras.get_intrinsics(dive.camera_id)

                new_axis, new_pos, n_pts, skipped = await compute_extrinsics(
                    client, dive, slate, labels, intrinsics
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  FAIL: {type(e).__name__}: {e}")
                results.append(
                    {"dive_id": dive.id, "status": "compute_fail", "error": repr(e)}
                )
                continue

            ang = axis_angle_deg(new_axis, stored.laser_axis)
            pos_l2 = float(np.linalg.norm(new_pos - np.asarray(stored.laser_position, dtype=float)))
            flipped = ang > 90.0
            within = (
                ang < AXIS_ANGLE_TOLERANCE_DEG
                and pos_l2 < POSITION_L2_TOLERANCE_M
                and not flipped
            )

            print(
                f"  axis_angle_deg={ang:.4f}  "
                f"position_l2_m={pos_l2:.6f}  "
                f"flipped={flipped}  "
                f"within_tol={within}  "
                f"n_points={n_pts}"
                + (f"  per_label_skips={len(skipped)}" if skipped else "")
            )
            results.append(
                {
                    "dive_id": dive.id,
                    "camera_id": dive.camera_id,
                    "status": "ok",
                    "axis_angle_deg": ang,
                    "position_l2_m": pos_l2,
                    "axis_180_flipped": bool(flipped),
                    "within_tolerance": bool(within),
                    "n_laser_points": n_pts,
                    "n_skipped_labels": len(skipped),
                    "new_axis": new_axis.tolist(),
                    "new_position": new_pos.tolist(),
                    "stored_axis": np.asarray(stored.laser_axis, dtype=float).tolist(),
                    "stored_position": np.asarray(stored.laser_position, dtype=float).tolist(),
                }
            )

        ok = [r for r in results if r["status"] == "ok"]
        within = [r for r in ok if r["within_tolerance"]]
        flipped = [r for r in ok if r["axis_180_flipped"]]
        print()
        print("=" * 64)
        print(
            f"SUMMARY  sample={len(sample)}  computed={len(ok)}  "
            f"within_tol={len(within)}/{len(ok)}  axis_180_flipped={len(flipped)}"
        )
        if ok:
            angs = [r["axis_angle_deg"] for r in ok]
            l2s = [r["position_l2_m"] for r in ok]
            print(
                f"  axis_angle_deg     min={min(angs):.4f}  "
                f"median={np.median(angs):.4f}  max={max(angs):.4f}"
            )
            print(
                f"  position_l2_m      min={min(l2s):.6f}  "
                f"median={np.median(l2s):.6f}  max={max(l2s):.6f}"
            )
        print(
            f"  tolerance: axis<{AXIS_ANGLE_TOLERANCE_DEG}deg  "
            f"position<{POSITION_L2_TOLERANCE_M}m  no 180-flips"
        )
        print("=" * 64)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(__file__).parent / f"validate_stage13_refactor_{ts}.json"
        out.write_text(
            json.dumps(
                {
                    "tolerance": {
                        "axis_angle_deg": AXIS_ANGLE_TOLERANCE_DEG,
                        "position_l2_m": POSITION_L2_TOLERANCE_M,
                    },
                    "sample_size": SAMPLE_SIZE,
                    "random_seed": RANDOM_SEED,
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"\nResults written to {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

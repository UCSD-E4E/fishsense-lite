"""Dry-run stage13 against the uncalibrated HIGH-priority dives in production.

Same computation as stage13_perform_laser_calibration.ipynb, but does NOT call
put_laser_extrinsics. Filters to dives that DON'T yet have laser_extrinsics
(i.e., the actual stage13 target cohort), computes new extrinsics, and prints
+ dumps the values for human review before any writes hit the API.

Phase 1 (validate_stage13_refactor.py) already verified value-equivalence vs
the deployed pre-refactor implementation across calibrated dives. This is the
final review gate before Phase 2 writes.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fishsense_api_sdk.client import Client
from fishsense_core.laser import calibrate_laser as _calibrate_laser
from fishsense_core.world_point import WorldPointHandler

from fishsense_data_processing_workflow_worker.config import settings

INCH_TO_M = 0.0254


def image_coordinate_to_projected_point(
    image_point: np.ndarray, k_inv: np.ndarray
) -> np.ndarray:
    return WorldPointHandler(k_inv).project_image_point(image_point)


def get_normal_vector_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, 2]


def atanasov_calibration_method(ps: np.ndarray) -> np.ndarray:
    origin, orientation = _calibrate_laser(ps.astype(np.float32))
    return np.concatenate([orientation.astype(float), origin[:2].astype(float)])


async def compute_extrinsics(
    client: Client, dive, dive_slate, dive_slate_labels, camera_intrinsics
):
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
        without_calib = [d for d, le in zip(dives, extrinsics) if le is None]
        print(f"  ...without existing laser_extrinsics: {len(without_calib)}\n")
        if not without_calib:
            print("No uncalibrated HIGH dives. Nothing to dry-run.")
            return 0

        slates = await client.dive_slates.get()
        slates_by_id = {s.id: s for s in slates}

        results: list[dict] = []
        for i, dive in enumerate(without_calib, 1):
            print(
                f"[{i}/{len(without_calib)}] dive_id={dive.id} camera_id={dive.camera_id}"
            )
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

                axis, position, n_pts, skipped = await compute_extrinsics(
                    client, dive, slate, labels, intrinsics
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  FAIL: {type(e).__name__}: {e}")
                results.append(
                    {"dive_id": dive.id, "status": "compute_fail", "error": repr(e)}
                )
                continue

            axis_norm = float(np.linalg.norm(axis))
            pos_norm = float(np.linalg.norm(position))
            print(
                f"  laser_axis     = [{axis[0]:+.5f}, {axis[1]:+.5f}, {axis[2]:+.5f}]  "
                f"|axis|={axis_norm:.5f}"
            )
            print(
                f"  laser_position = [{position[0]:+.5f}, {position[1]:+.5f}, {position[2]:+.5f}] m  "
                f"|pos|={pos_norm*1000:.1f}mm"
            )
            print(
                f"  n_laser_points={n_pts}"
                + (f"  per_label_skips={len(skipped)}" if skipped else "")
            )
            results.append(
                {
                    "dive_id": dive.id,
                    "camera_id": dive.camera_id,
                    "status": "ok",
                    "laser_axis": axis.tolist(),
                    "laser_axis_magnitude": axis_norm,
                    "laser_position_m": position.tolist(),
                    "laser_position_magnitude_m": pos_norm,
                    "n_laser_points": n_pts,
                    "n_skipped_labels": len(skipped),
                    "skipped_labels": skipped,
                }
            )

        ok = [r for r in results if r["status"] == "ok"]
        print()
        print("=" * 64)
        print(
            f"DRY-RUN SUMMARY  total={len(without_calib)}  "
            f"computed={len(ok)}  failed={len(without_calib) - len(ok)}"
        )
        if ok:
            axis_norms = [r["laser_axis_magnitude"] for r in ok]
            pos_norms = [r["laser_position_magnitude_m"] for r in ok]
            print(
                f"  |axis|     min={min(axis_norms):.5f}  max={max(axis_norms):.5f}  "
                f"(expect ~1.0)"
            )
            print(
                f"  |position| min={min(pos_norms) * 1000:6.1f}mm  "
                f"max={max(pos_norms) * 1000:6.1f}mm"
            )
        print("NO WRITES PERFORMED. Review values; run stage13 for real to persist.")
        print("=" * 64)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(__file__).parent / f"dry_run_stage13_{ts}.json"
        out.write_text(json.dumps({"dives": results}, indent=2))
        print(f"\nDetailed results written to {out}")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

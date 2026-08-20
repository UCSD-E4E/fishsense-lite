"""Shared activity helpers for the data-processing worker."""

from dataclasses import dataclass

from fishsense_api_sdk.client import Client

from fishsense_data_processing_workflow_worker.config import settings


def get_fs_client() -> Client:
    """Build a fishsense-api SDK client from the worker's dynaconf settings.

    Centralized so per-activity modules don't each construct one — when
    we eventually add timeouts / retries / custom auth, only this helper
    has to change.
    """
    return Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    )


@dataclass(frozen=True)
class DiveCalibrationContext:
    """Everything needed to turn a laser dot on an image of this dive into a
    position in space: the dive, its camera's intrinsics, and the
    `LaserExtrinsics` the dive resolves to."""

    dive: object
    camera_intrinsics: object
    laser_extrinsics: object


async def load_dive_calibration_context(fs, dive_id: int) -> DiveCalibrationContext:
    """Fetch the three prerequisites for projecting a laser dot, or raise.

    Shared by stage 14 and the laser-depth stage, which need exactly the same
    preconditions and must agree on what "this dive is not ready" means —
    they had drifted apart into two copies of this preamble before
    `duplicate-code` flagged it.

    Raises `ValueError` rather than returning None: every caller's cohort
    selector already filters on all three, so a miss here means the selector
    and the activity disagree, which is the failure mode that leaves a dive in
    a cohort forever. Failing loud makes it a visible workflow failure instead
    of a silent no-op.
    """
    dive = await fs.dives.get(dive_id=dive_id)
    if dive is None:
        raise ValueError(f"dive_id={dive_id} not found")
    if dive.camera_id is None:
        raise ValueError(f"dive_id={dive_id} has no camera_id")

    camera_intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
    if camera_intrinsics is None:
        raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

    # Resolves own-then-borrowed via `Dive.calibration_dive_id`, so a
    # fish-only dive transparently gets its sibling's rig calibration.
    laser_extrinsics = await fs.dives.get_laser_extrinsics(dive_id)
    if laser_extrinsics is None:
        raise ValueError(
            f"dive_id={dive_id} has no laser_extrinsics; "
            "run perform_laser_calibration_activity first"
        )

    return DiveCalibrationContext(
        dive=dive,
        camera_intrinsics=camera_intrinsics,
        laser_extrinsics=laser_extrinsics,
    )

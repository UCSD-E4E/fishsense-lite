"""Activity to resolve everything checkerboard calibration needs for a dive.

Returns a fully-populated `PerformCheckerboardCalibrationInput`: the camera's
intrinsics, the board's measured geometry, and one entry per frame that
carries a live laser dot. The data-worker child then makes no SDK calls and
takes no decisions — the standing cross-worker split, and the reason a replay
cannot pick up a different square size than the run was dispatched with.

The image filter mirrors the cohort selector exactly, per CLAUDE.md: live
(non-superseded, x/y populated) laser label, canonical image. A resolver that
selected differently from its cohort is the failure that stages a dive's raw
`.ORF`s from the NAS every hour and dispatches nothing.

**Nothing here looks for a board.** The dot is the pipeline's own product and a
row can be read; the board is pixels, and finding it means decoding the frame,
which is the data-worker's job and the reason the child exists. So this
activity's answer is "here are the frames worth *trying*", never "here are the
frames that will work".
"""

from __future__ import annotations

from fishsense_shared import (
    CheckerboardCalibrationImage,
    PerformCheckerboardCalibrationInput,
)
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client


def _has_live_dot(label) -> bool:
    """The same predicate `get_laser_label` and the cohort selector use.

    Nothing about `completed`, deliberately: a populate-seeded placeholder is
    excluded by its NULL x/y, not by its completion state — which is what the
    api-side `select_next_for_checkerboard_laser_calibration` counts.
    """
    return bool(not label.superseded and label.x is not None and label.y is not None)


@activity.defn
async def resolve_checkerboard_calibration_inputs_activity(
    dive_id: int,
) -> PerformCheckerboardCalibrationInput:
    activity.logger.info(
        "resolving checkerboard calibration inputs dive_id=%d", dive_id
    )
    async with get_fs_client() as fs:
        dive = await fs.dives.get(dive_id=dive_id)
        if dive is None:
            raise ValueError(f"dive_id={dive_id} not found")
        if dive.camera_id is None:
            raise ValueError(f"dive_id={dive_id} has no camera_id")
        if dive.calibration_target_id is None:
            raise ValueError(f"dive_id={dive_id} has no calibration_target_id")

        intrinsics = await fs.cameras.get_intrinsics(dive.camera_id)
        if intrinsics is None:
            raise ValueError(f"camera_id={dive.camera_id} has no intrinsics")

        targets = await fs.calibration_targets.get() or []
        target = next((t for t in targets if t.id == dive.calibration_target_id), None)
        if target is None:
            raise ValueError(
                f"dive_id={dive_id}: calibration_target_id="
                f"{dive.calibration_target_id} not found"
            )

        images = await fs.images.get(dive_id=dive_id) or []
        # Canonical frames only — the same physical frames live under several
        # dive rows (two of the 2023.08.18 dives are wholly or partly duplicate
        # content), and every cohort selector gates on this.
        checksum_by_id = {
            image.id: image.checksum for image in images if image.is_canonical
        }

        laser_labels = await fs.labels.get_laser_labels(dive_id) or []
        frames = [
            CheckerboardCalibrationImage(
                image_id=label.image_id,
                checksum=checksum_by_id[label.image_id],
                laser_x=float(label.x),
                laser_y=float(label.y),
            )
            for label in laser_labels
            if _has_live_dot(label) and label.image_id in checksum_by_id
        ]

        activity.logger.info(
            "resolved checkerboard calibration inputs dive_id=%d target=%r "
            "board=%dx%d pitch=%.5fm frames=%d",
            dive_id,
            target.name,
            target.rows,
            target.cols,
            target.square_size_m,
            len(frames),
        )
        return PerformCheckerboardCalibrationInput(
            dive_id=dive_id,
            camera_id=dive.camera_id,
            camera_matrix=intrinsics.camera_matrix.tolist(),
            distortion_coefficients=intrinsics.distortion_coefficients.tolist(),
            target_rows=target.rows,
            target_cols=target.cols,
            square_size_m=float(target.square_size_m),
            images=frames,
        )

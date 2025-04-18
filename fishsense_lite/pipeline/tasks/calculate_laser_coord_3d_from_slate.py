import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fishsense_common.pipeline.decorators import task
from fishsense_common.pipeline.status import error, ok
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.library.homography import viz2d
from pyfishsensedev.plane_detector.slate_detector import SlateDetector
from skimage.util import img_as_ubyte


@task(output_name="laser_coord_3d")
def calculate_laser_coord_3d_from_slate(
    input_file: Path,
    img: np.ndarray[float],
    pdf: Pdf,
    laser_image_coords: np.ndarray[int],
    lens_calibration: LensCalibration,
    device: str,
    try_multiple_slate_rotations: bool,
    debug_path: Path,
) -> np.ndarray[float]:
    slate_detector = SlateDetector(
        img_as_ubyte(img),
        pdf,
        lens_calibration,
        device,
        try_multiple_slate_rotations=try_multiple_slate_rotations,
    )
    if not slate_detector.is_valid():
        return error("INVALID_STATE")

    template_matches, image_matches = slate_detector._get_template_matches()

    rotation, _ = slate_detector._get_body_to_camera_space_transform()

    if debug_path is not None:
        hash = hashlib.md5(input_file.read_bytes()).hexdigest()
        png_name = f"{hash}.png"

        plt.clf()
        viz2d.plot_images([slate_detector.pdf.image, img[:, :, ::-1]])
        viz2d.plot_matches(template_matches, image_matches, color="lime", lw=0.2)
        viz2d.add_text(
            0,
            f"{len(template_matches)} matches; up vector: {np.round(rotation @ np.array([0, 1, 0]) * 100.0) / 100.0}",
            fs=20,
        )
        plt.savefig((debug_path / f"matches_{png_name}"))

    laser_coord_3d = slate_detector.project_point_onto_plane_camera_space(
        laser_image_coords
    )

    if np.any(np.isnan(laser_coord_3d)):
        return error("INVALID_LASER_3D_COORDINATES")

    return ok(laser_coord_3d)

import hashlib
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from fishsense_common.pipeline.decorators import task
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
    debug_path: Path,
) -> np.ndarray[float]:
    if (
        input_file is None
        or img is None
        or pdf is None
        or laser_image_coords is None
        or lens_calibration is None
    ):
        return None

    if debug_path is not None:
        hash = hashlib.md5(input_file.read_bytes()).hexdigest()
        png_name = f"{hash}.png"

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(pdf.image)
        axarr[1].imshow(img)
        f.savefig((debug_path / f"prematch_{png_name}"))
        f.show()

        cv2.imwrite((debug_path / f"img_{png_name}").as_posix(), img_as_ubyte(img))

    slate_detector = SlateDetector(img_as_ubyte(img), pdf)
    if not slate_detector.is_valid():
        return None

    template_matches, image_matches = slate_detector._get_template_matches()

    rotation, _ = slate_detector._get_body_to_camera_space_transform(
        lens_calibration.camera_matrix
    )

    if debug_path is not None:
        plt.clf()
        viz2d.plot_images([pdf.image, img])
        viz2d.plot_matches(template_matches, image_matches, color="lime", lw=0.2)
        viz2d.add_text(
            0,
            f"{len(template_matches)} matches; up vector: {rotation @ np.array([0, 1, 0])}",
            fs=20,
        )
        plt.savefig((debug_path / f"matches_{png_name}"))

    laser_coord_3d = slate_detector.project_point_onto_plane_camera_space(
        laser_image_coords,
        lens_calibration.camera_matrix,
        lens_calibration.inverted_camera_matrix,
    )

    if np.any(np.isnan(laser_coord_3d)):
        return None

    return laser_coord_3d

import hashlib
from pathlib import Path

import cv2
import numpy as np
from fishsense_common.pipeline.decorators import task
from fishsense_common.pipeline.status import error, ok
from pyfishsensedev.laser.laser_detector import LaserDetector
from skimage.util import img_as_ubyte


@task(output_name="laser_image_coords")
def detect_laser(
    input_file: Path,
    img: np.ndarray[float],
    laser_detector: LaserDetector,
    debug_path: Path,
) -> np.ndarray:
    laser_image_coords = laser_detector.find_laser(img)

    if laser_image_coords is None:
        return error("INVALID_LASER_IMAGE_COORDINATES")

    if debug_path is not None:
        hash = hashlib.md5(input_file.read_bytes()).hexdigest()
        png_name = f"{hash}.png"

        laser_detection_path = debug_path / f"detection_{png_name}"
        if laser_detection_path.exists():
            laser_detection_path.unlink()

        laser_detection = cv2.circle(
            img_as_ubyte(img),
            np.round(laser_image_coords).astype(int),
            radius=5,
            color=(0, 255, 255),
            thickness=-1,
        )
        cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)

    return ok(laser_image_coords)

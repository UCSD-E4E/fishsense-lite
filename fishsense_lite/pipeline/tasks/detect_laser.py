from pathlib import Path

import cv2
import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.laser.laser_detector import LaserDetector
from skimage.util import img_as_ubyte


@task(output_name="laser_image_coords")
def detect_laser(
    input_file: Path,
    img: np.ndarray[float],
    laser_detector: LaserDetector,
    debug_path: Path,
) -> np.ndarray:
    if img is None or laser_detector is None:
        return None

    laser_image_coords = laser_detector.find_laser(img)

    if laser_image_coords is None:
        return None

    if debug_path is not None:
        png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")
        laser_detection_path = debug_path / f"detection_{png_name}"
        if laser_detection_path.exists():
            laser_detection_path.unlink()

        laser_detection = cv2.circle(
            img_as_ubyte(img),
            np.round(laser_image_coords).astype(int),
            radius=5,
            color=(0, 255, 0),
            thickness=-1,
        )
        cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)

    return laser_image_coords

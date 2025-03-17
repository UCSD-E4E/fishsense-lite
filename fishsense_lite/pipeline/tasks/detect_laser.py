import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.laser.laser_detector import LaserDetector


@task(output_name="laser_image_coords")
def detect_laser(img: np.ndarray[float], laser_detector: LaserDetector) -> np.ndarray:
    return laser_detector.find_laser(img)

import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.laser.laser_detector import LaserDetector


@task(output_name="laser_image_coords")
def detect_laser(img: np.ndarray[float], laser_detector: LaserDetector) -> np.ndarray:
    return laser_detector.find_laser(img)

    # laser_detection_path = debug_path / f"detection_{png_name}"
    # if laser_detection_path.exists():
    #     laser_detection_path.unlink()

    # laser_detection = cv2.circle(
    #     image_dark.copy(),
    #     np.round(laser_image_coord).astype(int),
    #     radius=5,
    #     color=(0, 255, 0),
    #     thickness=-1,
    # )
    # cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)

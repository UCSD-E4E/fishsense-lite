import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.depth_map import LaserDepthMap


@task(output_name="length")
def calculate_length(
    img: np.ndarray[float],
    laser_image_coords: np.ndarray,
    left_point: np.ndarray,
    right_point: np.ndarray,
    lens_calibration: LensCalibration,
    laser_calibration: LaserCalibration,
) -> float:
    depth_map = LaserDepthMap(laser_image_coords, lens_calibration, laser_calibration)

    image_height, image_width, _ = img.shape

    left_coord3d = depth_map.get_camera_space_point(
        left_point, image_width, image_height, lens_calibration
    )
    right_coord3d = depth_map.get_camera_space_point(
        right_point, image_width, image_height, lens_calibration
    )

    return np.linalg.norm(left_coord3d - right_coord3d)

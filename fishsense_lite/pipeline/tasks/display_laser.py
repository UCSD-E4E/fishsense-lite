import cv2
import numpy as np
from fishsense_common.pipeline.decorators import task


@task(output_name="laser_detection_img")
def display_laser(
    img: np.ndarray[float], laser_image_coords: np.ndarray[int]
) -> np.ndarray[float]:
    if img is None or laser_image_coords is None:
        return None

    return cv2.circle(
        img,
        np.round(laser_image_coords).astype(int),
        radius=5,
        color=(255, 0, 0),
        thickness=-1,
    )

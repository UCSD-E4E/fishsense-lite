import cv2
import numpy as np
from fishsense_common.pipeline.decorators import task
from skimage.util import img_as_float, img_as_ubyte


@task(output_name="laser_detection_img")
def display_laser(
    img: np.ndarray[float], laser_image_coords: np.ndarray[int]
) -> np.ndarray[float]:
    if img is None or laser_image_coords is None:
        return None

    return img_as_float(
        cv2.circle(
            img_as_ubyte(img),
            np.round(laser_image_coords).astype(int),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )
    )

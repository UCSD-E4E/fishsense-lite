import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image import ImageRectifier


@task(output_name="img")
def image_rectifier(
    img: np.ndarray[float], lens_calibration: LensCalibration
) -> np.ndarray[float]:
    if img is None:
        return None

    image_rectifier = ImageRectifier(lens_calibration)

    return image_rectifier.rectify(img)

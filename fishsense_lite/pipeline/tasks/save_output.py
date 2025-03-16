from pathlib import Path

import cv2
import numpy as np
from fishsense_common.pipeline.decorators import task
from skimage.util import img_as_ubyte, img_as_uint

from fishsense_lite.utils import get_output_file


@task()
def save_output(
    img: np.ndarray[float], input_file: Path, root: Path, output: Path, format: str
):
    if img is None:
        return None

    output_file = get_output_file(input_file, root, output, format)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "png":
        img = img_as_uint(img)
    elif format.lower() == "jpg":
        img = img_as_ubyte(img)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(output_file.absolute().as_posix(), img)

from pathlib import Path

import numpy as np
from fishsense_common.pipeline.decorators import task
from fishsense_common.pipeline.status import error, ok
from pyaqua3ddev.image.image_processors import RawProcessor
from skimage.util import img_as_float


@task(output_name="img")
def process_raw(input_file: Path) -> np.ndarray[float]:
    if input_file is None or input_file.exists() is False:
        return error("INVALID_INPUT_FILE")

    raw_processor = RawProcessor()

    try:
        return ok(img_as_float(raw_processor.process(input_file)))
    except:
        return error("INVALID_RAW_FILE")

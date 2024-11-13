import importlib
import random
import string

import numpy as np

from fishsense_lite.commands.label_studio_models.data import Data


class LaserValue:
    def __init__(self, x: float, y: float, width: int, height: int):
        self.x = x / float(width) * 100
        self.y = y / float(height) * 100
        self.width = 0.25
        self.keypointlabels = ["Red Laser"]


class LaserResult:
    def __init__(self, laser_image_coord: np.ndarray, width: int, height: int):
        self.original_width = width
        self.original_height = height
        self.image_rotation = 0
        self.value = LaserValue(
            laser_image_coord[0], laser_image_coord[1], width, height
        )

        letters_and_numbers = string.ascii_letters + string.digits

        self.id = "".join(random.choice(letters_and_numbers) for _ in range(10))
        self.from_name = "kp-1"
        self.to_name = "img-1"
        self.type = "keypointlabels"


class LaserPrediction:
    def __init__(
        self, laser_image_coord: np.ndarray, width: int, height: int, model_name: str
    ):
        self.model_version = (
            f"{model_name}.{importlib.metadata.version("pyfishsensedev")}"
        )
        self.result = [LaserResult(laser_image_coord, width, height)]


class LaserLabelStudioJSON:
    def __init__(
        self,
        prefix: str,
        img: str,
        laser_image_coord: np.ndarray,
        width: int,
        height: int,
        model_name: str,
    ):
        self.data = Data(prefix, img)
        self.predictions = (
            [LaserPrediction(laser_image_coord, width, height, model_name)]
            if laser_image_coord is not None
            else []
        )

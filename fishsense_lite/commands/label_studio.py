"""Module which represents the FishSense Lite Label Studio CLI."""

import importlib
import json
import random
import string
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import ray
import torch
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.image.image_processors import RawProcessor
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector

from fishsense_lite.utils import get_output_file, get_root, uint16_2_uint8


class Data:
    def __init__(self, img: str):
        self.img = img


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

        self.id = "".join(random.choice(letters_and_numbers) for i in range(10))
        self.from_name = "kp-1"
        self.to_name = "img-1"
        self.type = "keypointlabels"


class LaserPrediction:
    def __init__(self, laser_image_coord: np.ndarray, width: int, height: int):
        self.model_version = importlib.metadata.version("fishsense_lite")
        self.result = [LaserResult(laser_image_coord, width, height)]


class LaserLabelStudioJSON:
    def __init__(
        self, img: str, laser_image_coord: np.ndarray, width: int, height: int
    ):
        self.data = Data(img)
        self.predictions = (
            [LaserPrediction(laser_image_coord, width, height)]
            if laser_image_coord is not None
            else []
        )


@ray.remote(num_gpus=0.25)
def execute_laser(
    input_file: Path,
    lens_calibration: LensCalibration,
    estimated_laser_calibration: LaserCalibration,
    root: Path,
    output: Path,
    prefix: str,
    overwrite: bool,
) -> Tuple[np.ndarray, int, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_file = get_output_file(input_file, root, output, "jpg")
    json_file = output_file.with_suffix(".json")

    if output_file.exists() and not overwrite:
        return None, None, None

    dark_raw_processor = RawProcessor(enable_histogram_equalization=False)
    image_dark = uint16_2_uint8(dark_raw_processor.load_and_process(input_file))

    height, width, _ = image_dark.shape

    image_rectifier = ImageRectifier(lens_calibration)
    image_dark = image_rectifier.rectify(image_dark)

    laser_detector = NNLaserDetector(
        lens_calibration, estimated_laser_calibration, device
    )
    laser_image_coord = laser_detector.find_laser(image_dark)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_file.absolute().as_posix(), image_dark)

    json_objects = LaserLabelStudioJSON(
        f"{prefix}{output_file.relative_to(output.absolute()).as_posix()}",
        laser_image_coord,
        width,
        height,
    )

    with open(json_file, "w") as f:
        f.write(json.dumps(json_objects, default=vars))


class LabelStudioCommand(Command):
    """Command which represents the FishSense Lite Label Studio CLI."""

    @property
    def name(self) -> str:
        return "label-studio"

    @property
    def description(self) -> str:
        return "Outputs data in a format for comsuption with Label Studio."

    @property
    @argument("data", required=True, help="A glob that represents the data to process.")
    def data(self) -> List[str]:
        return self.__data

    @data.setter
    def data(self, value: List[str]):
        self.__data = value

    @property
    @argument(
        "--laser-position",
        short_name="-p",
        nargs=3,
        required=True,
        help="The laser position in centimeter inputed as x y z for the FishSense Lite product line.",
    )
    def laser_position(self) -> List[int]:
        return self.__laser_position

    @laser_position.setter
    def laser_position(self, value: List[int]):
        self.__laser_position = value

    @property
    @argument(
        "--laser-axis",
        short_name="-a",
        nargs=3,
        required=True,
        help="The laser axis unit vector inputed as x y z for the FishSense Lite product line.",
    )
    def laser_axis(self) -> List[float]:
        return self.__laser_axis

    @laser_axis.setter
    def laser_axis(self, value: List[float]):
        self.__laser_axis = value

    @property
    @argument(
        "--lens-calibration",
        short_name="-l",
        required=True,
        help="Lens calibration package for the FishSense Lite.",
    )
    def lens_calibration(self) -> str:
        return self.__lens_calibration

    @lens_calibration.setter
    def lens_calibration(self, value: str):
        self.__lens_calibration = value

    @property
    @argument(
        "--output",
        short_name="-o",
        required=True,
        help="The path to store the resulting database.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument(
        "--prefix",
        short_name="-p",
        required=True,
        help="The prefix to add to the output json file.",
    )
    def prefix(self) -> str:
        return self.__prefix

    @prefix.setter
    def prefix(self, value: str):
        self.__prefix = value

    @property
    @argument("--overwrite", flag=True, help="Overwrite the calibration if it exists.")
    def overwrite(self) -> bool:
        return self.__overwrite

    @overwrite.setter
    def overwrite(self, value: bool):
        self.__overwrite = value

    def __init__(self):
        super().__init__()

        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_position: List[int] = None
        self.__laser_axis: List[float] = None
        self.__output_path: str = None
        self.__prefix: str = None
        self.__overwrite: bool = None

    def __call__(self):
        self.init_ray()

        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}
        root = get_root(files)

        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        estimated_laser_calibration = LaserCalibration(
            np.array(self.laser_axis), np.array(self.laser_position)
        )

        output = Path(self.output_path)

        self.__build_laser_json(
            files, lens_calibration, estimated_laser_calibration, root, output
        )

    def __build_laser_json(
        self,
        files: List[Path],
        lens_calibration: LensCalibration,
        estimated_laser_calibration: LaserCalibration,
        root: Path,
        output: Path,
    ):
        output = output / "laser"
        output.mkdir(parents=True, exist_ok=True)

        laser_json_path = output / "label_studio.json"
        if laser_json_path.exists() and not self.overwrite:
            return

        futures = [
            execute_laser.remote(
                f,
                lens_calibration,
                estimated_laser_calibration,
                root,
                output,
                self.prefix,
                self.overwrite,
            )
            for f in files
        ]

        list(self.tqdm(futures, total=len(files)))

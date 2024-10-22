from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import ray
import torch
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.depth_map import LaserDepthMap
from pyfishsensedev.image import ImageRectifier, RawProcessor
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector
from pyfishsensedev.points_of_interest.fish import FishPointsOfInterestDetector
from pyfishsensedev.segmentation.fish.fish_segmentation_fishial_pytorch import (
    FishSegmentationFishialPyTorch,
)
from tqdm import tqdm

from fishsense_lite.database import Database
from fishsense_lite.result_status import ResultStatus


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def uint16_2_double(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 65535


def uint16_2_uint8(img: np.ndarray) -> np.ndarray:
    return (uint16_2_double(img) * 255).astype(np.uint8)


@ray.remote(num_gpus=0.25)
def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    laser_calibration: LaserCalibration,
) -> Tuple[Path, ResultStatus, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_path = Path(".debug") / "process"
    debug_path.mkdir(exist_ok=True, parents=True)

    raw_processor_hist_eq = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    image_rectifier = ImageRectifier(lens_calibration)

    img = raw_processor_hist_eq.load_and_process(input_file)
    img_dark = raw_processor.load_and_process(input_file)

    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = uint16_2_uint8(img)
    img_dark8 = uint16_2_uint8(img_dark)

    debug_output = img8.copy()

    laser_detector = NNLaserDetector(lens_calibration, laser_calibration, device)
    laser_coords = laser_detector.find_laser(img_dark8)

    if laser_coords is None:
        debug_file = (
            debug_path
            / f"{input_file.name[:-4]}_{ResultStatus.FAILED_LASER_COORDS}.png"
        )
        cv2.imwrite(debug_file.absolute().as_posix(), debug_output)
        return input_file, ResultStatus.FAILED_LASER_COORDS, None

    laser_coords_int = np.round(laser_coords).astype(int)
    debug_output = cv2.circle(
        debug_output,
        laser_coords_int,
        radius=5,
        color=(0, 255, 0),
        thickness=-1,
    )

    fish_segmentation_inference = FishSegmentationFishialPyTorch(device)
    segmentations: np.ndarray = fish_segmentation_inference.inference(img8)

    if segmentations.sum() == 0:
        debug_file = (
            debug_path
            / f"{input_file.name[:-4]}_{ResultStatus.FAILED_SEGMENTATION}.png"
        )
        cv2.imwrite(debug_file.absolute().as_posix(), debug_output)
        return input_file, ResultStatus.FAILED_SEGMENTATION, None

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[laser_coords_int[1], laser_coords_int[0]]] = (
        True
    )

    if segmentations[laser_coords_int[1], laser_coords_int[0]] == 0 or mask.sum() == 0:
        debug_file = (
            debug_path
            / f"{input_file.name[:-4]}_{ResultStatus.FAILED_LASER_SEGMENTATION_INTERSECTION}.png"
        )
        cv2.imwrite(debug_file.absolute().as_posix(), debug_output)
        return input_file, ResultStatus.FAILED_LASER_SEGMENTATION_INTERSECTION, None

    debug_output[mask == False, :] = (debug_output[mask == False, :] * 0.5).astype(
        np.uint8
    )

    fish_head_tail_detector = FishPointsOfInterestDetector()
    tail_coord, head_coord = fish_head_tail_detector.find_points_of_interest(mask)

    debug_output = cv2.circle(
        debug_output,
        np.round(tail_coord).astype(int),
        radius=25,
        color=(0, 0, 255),
        thickness=-1,
    )
    debug_output = cv2.circle(
        debug_output,
        np.round(head_coord).astype(int),
        radius=25,
        color=(255, 0, 0),
        thickness=-1,
    )

    depth_map = LaserDepthMap(laser_coords, lens_calibration, laser_calibration)

    image_height, image_width, _ = img.shape

    tail_coord3d = depth_map.get_camera_space_point(
        tail_coord, image_width, image_height, lens_calibration
    )
    head_coord3d = depth_map.get_camera_space_point(
        head_coord, image_width, image_height, lens_calibration
    )
    length = np.linalg.norm(tail_coord3d - head_coord3d)

    debug_file = debug_path / f"{input_file.name[:-4]}_{ResultStatus.SUCCESS}.png"
    cv2.imwrite(debug_file.absolute().as_posix(), debug_output)

    return input_file, ResultStatus.SUCCESS, length


class Process(Command):
    @property
    def name(self) -> str:
        return "process"

    @property
    def description(self) -> str:
        return "Process data from the FishSense Lite product line."

    @property
    @argument("data", required=True, help="A glob that represents the data to process.")
    def data(self) -> List[str]:
        return self.__data

    @data.setter
    def data(self, value: List[str]):
        self.__data = value

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
        "--laser-calibration",
        short_name="-k",
        required=True,
        help="Laser calibration package for the FishSense Lite.",
    )
    def laser_calibration(self) -> str:
        return self.__laser_calibration

    @laser_calibration.setter
    def laser_calibration(self, value: str):
        self.__laser_calibration = value

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
    @argument("--overwrite", flag=True, help="Overwrite the database if it exists.")
    def overwrite(self) -> bool:
        return self.__overwrite

    @overwrite.setter
    def overwrite(self, value: bool):
        self.__overwrite = value

    def __init__(self):
        super().__init__()

        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_calibration: str = None
        self.__output_path: str = None
        self.__overwrite: bool = None

    def __call__(self):
        self.init_ray()

        with Database(Path(self.output_path)) as database:
            files = {Path(f) for g in self.data for f in glob(g, recursive=True)}

            if not self.overwrite:
                prev_files = database.get_files()
                files.difference_update(prev_files)

            lens_calibration = LensCalibration()
            laser_calibration = LaserCalibration()

            lens_calibration.load(Path(self.lens_calibration))
            laser_calibration.load(Path(self.laser_calibration))

            futures = [
                execute.remote(f, lens_calibration, laser_calibration) for f in files
            ]

            # for file, result_status, length in tqdm(
            #     (execute(f, lens_calibration, laser_calibration) for f in files),
            #     total=len(files),
            # ):
            for file, result_status, length in tqdm(
                to_iterator(futures), total=len(files)
            ):
                database.insert_data(file, result_status, length)

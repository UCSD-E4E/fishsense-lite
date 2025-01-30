from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import fishsense_common.ray as ray
import numpy as np
import torch
from fishsense_common.pluggable_cli import Command, argument
from pyaqua3ddev.image.image_processors import RawProcessor
from pyaqua3ddev.laser.single_laser.label_studio_laser_detector import (
    LabelStudioLaserDetector,
)
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.depth_map import LaserDepthMap
from pyfishsensedev.image import ImageRectifier
from pyfishsensedev.points_of_interest.fish import FishPointsOfInterestDetector
from pyfishsensedev.points_of_interest.fish.fish_label_studio_points_of_interest_detector import (
    FishLabelStudioPointsOfInterestDetector,
)
from pyfishsensedev.segmentation.fish.fish_segmentation_fishial_pytorch import (
    FishSegmentationFishialPyTorch,
)
from tqdm import tqdm

from fishsense_lite.database import Database
from fishsense_lite.result_status import ResultStatus
from fishsense_lite.utils import uint16_2_uint8


def find_head_tail(
    img: np.ndarray[np.uint8],
    laser_coords: np.ndarray[np.uint8],
    device: str,
    input_file: Path,
    debug_output: np.ndarray[np.uint8],
    debug_path: Path,
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    fish_segmentation_inference = FishSegmentationFishialPyTorch(device)
    segmentations: np.ndarray = fish_segmentation_inference.inference(img)

    if segmentations.sum() == 0:
        debug_file = (
            debug_path
            / f"{input_file.name[:-4]}_{ResultStatus.FAILED_SEGMENTATION}.png"
        )
        cv2.imwrite(debug_file.absolute().as_posix(), debug_output)
        return input_file, ResultStatus.FAILED_SEGMENTATION, None

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[laser_coords[1], laser_coords[0]]] = True

    if segmentations[laser_coords[1], laser_coords[0]] == 0 or mask.sum() == 0:
        debug_file = (
            debug_path
            / f"{input_file.name[:-4]}_{ResultStatus.FAILED_LASER_SEGMENTATION_INTERSECTION}.png"
        )
        cv2.imwrite(debug_file.absolute().as_posix(), debug_output)
        return input_file, ResultStatus.FAILED_LASER_SEGMENTATION_INTERSECTION, None

    laser_coords[mask == False, :] = (laser_coords[mask == False, :] * 0.5).astype(
        np.uint8
    )

    fish_head_tail_detector = FishPointsOfInterestDetector()
    head_coord, tail_coord = fish_head_tail_detector.find_points_of_interest(mask)

    return (head_coord, tail_coord)


# @ray.remote(vram_mb=1536)
def execute(
    input_file: Path,
    laser_label_studio_json_path: Path | None,
    headtail_label_studio_json_path: Path | None,
    lens_calibration: LensCalibration,
    laser_calibration: LaserCalibration,
    debug_root: Path,
) -> Tuple[Path, ResultStatus, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_path = debug_root / "process"
    debug_path.mkdir(exist_ok=True, parents=True)

    raw_processor = RawProcessor()
    image_rectifier = ImageRectifier(lens_calibration)

    img = raw_processor.process(input_file)
    img = image_rectifier.rectify(img)
    img8 = uint16_2_uint8(img)

    debug_output = img8.copy()

    if laser_label_studio_json_path is not None:
        laser_detector = LabelStudioLaserDetector(
            input_file, laser_label_studio_json_path
        )
    else:
        raise NotImplementedError

    try:
        laser_coords = laser_detector.find_laser(img8)
    except KeyError:
        laser_coords = None

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
        color=(255, 0, 0),
        thickness=-1,
    )

    if headtail_label_studio_json_path is not None:
        poi_detector = FishLabelStudioPointsOfInterestDetector(
            input_file, headtail_label_studio_json_path
        )

        try:
            head_coord, tail_coord = poi_detector.find_points_of_interest(None)
        except KeyError:
            head_coord, tail_coord = None, None
    else:
        head_coord, tail_coord = find_head_tail(
            img8, laser_coords_int, device, input_file, debug_output, debug_path
        )

    if head_coord is None or tail_coord is None:
        return input_file, ResultStatus.FAILED_SEGMENTATION, None

    debug_output = cv2.circle(
        debug_output,
        np.round(tail_coord).astype(int),
        radius=5,
        color=(0, 0, 255),
        thickness=-1,
    )
    debug_output = cv2.circle(
        debug_output,
        np.round(head_coord).astype(int),
        radius=5,
        color=(0, 255, 0),
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
        short_name="-m",
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
        "--laser-label-studio-json",
        short_name="-j",
        help="An export of JSON tasks from Label Studio",
    )
    def laser_label_studio_json(self) -> str:
        return self.__laser_label_studio_json

    @laser_label_studio_json.setter
    def laser_label_studio_json(self, value: str):
        self.__laser_label_studio_json = value

    @property
    @argument(
        "--headtail-label-studio-json",
        short_name="-k",
        help="An export of JSON tasks from Label Studio",
    )
    def headtail_label_studio_json(self) -> str:
        return self.__headtail_label_studio_json

    @headtail_label_studio_json.setter
    def headtail_label_studio_json(self, value: str):
        self.__headtail_label_studio_json = value

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

    @property
    @argument("--debug-path", help="Sets the debug path for storing debug images.")
    def debug_path(self) -> str:
        return self.__debug_path

    @debug_path.setter
    def debug_path(self, value: str):
        self.__debug_path = value

    def __init__(self):
        super().__init__()

        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_calibration: str = None
        self.__laser_label_studio_json: str = None
        self.__headtail_label_studio_json: str = None
        self.__output_path: str = None
        self.__overwrite: bool = None
        self.__debug_path: str = None

    def __call__(self):
        self.init_ray()

        if self.debug_path is None:
            self.debug_path = ".debug"

        debug_path = Path(self.debug_path)
        laser_label_studio_json_path = (
            Path(self.laser_label_studio_json)
            if self.laser_label_studio_json is not None
            else None
        )
        headtail_label_studio_json_path = (
            Path(self.headtail_label_studio_json)
            if self.headtail_label_studio_json is not None
            else None
        )

        with Database(Path(self.output_path)) as database:
            files = {Path(f) for g in self.data for f in glob(g, recursive=True)}

            if not self.overwrite:
                prev_files = database.get_files()
                files.difference_update(prev_files)

            lens_calibration = LensCalibration()
            laser_calibration = LaserCalibration()

            lens_calibration.load(Path(self.lens_calibration))
            laser_calibration.load(Path(self.laser_calibration))

            # futures = [
            #     execute.remote(
            #         f,
            #         laser_label_studio_json_path,
            #         headtail_label_studio_json_path,
            #         lens_calibration,
            #         laser_calibration,
            #         debug_path,
            #     )
            #     for f in files
            # ]

            for file, result_status, length in tqdm(
                (
                    execute(
                        f,
                        laser_label_studio_json_path,
                        headtail_label_studio_json_path,
                        lens_calibration,
                        laser_calibration,
                        debug_path,
                    )
                    for f in files
                ),
                total=len(files),
            ):
            # for file, result_status, length in self.tqdm(futures, total=len(files)):
                database.insert_data(file, result_status, length)

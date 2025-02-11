import json
from glob import glob
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import cv2
import fishsense_common.ray as ray
import numpy as np
from fishsense_common.pluggable_cli import Command, argument
from pyaqua3ddev.image.image_processors import RawProcessor
from pyaqua3ddev.laser.single_laser.label_studio_laser_detector import (
    LabelStudioLaserDetector,
)
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.depth_map import LaserDepthMap
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.plane_detector.checkerboard_detector import CheckerboardDetector

from fishsense_lite.utils import uint16_2_uint8


@ray.remote(vram_mb=1536)
def execute(
    input_file: Path,
    laser_label_studio_json_path: Path | None,
    lens_calibration: LensCalibration,
    rows: int,
    columns: int,
    square_size: float,
    debug_root: Path,
) -> np.ndarray | None:
    debug_path = debug_root / "calibration" / "laser"
    debug_path.mkdir(exist_ok=True, parents=True)

    png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")

    raw_processor = RawProcessor()

    try:
        image = uint16_2_uint8(raw_processor.process(input_file))
    except:
        return None

    image_rectifier = ImageRectifier(lens_calibration)
    image = image_rectifier.rectify(image)

    if laser_label_studio_json_path is not None:
        laser_detector = LabelStudioLaserDetector(
            input_file, laser_label_studio_json_path
        )
    else:
        raise NotImplementedError

    laser_image_coord = laser_detector.find_laser(image)
    if laser_image_coord is None:
        return None

    laser_detection_path = debug_path / f"detection_{png_name}"
    if laser_detection_path.exists():
        laser_detection_path.unlink()

    laser_detection = cv2.circle(
        image,
        np.round(laser_image_coord).astype(int),
        radius=5,
        color=(255, 0, 0),
        thickness=-1,
    )
    cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)

    checkerboard_detector = CheckerboardDetector(
        image, rows, columns, square_size * 10**-3
    )

    if not checkerboard_detector.is_valid():
        return None

    laser_coord_3d = checkerboard_detector.project_point_onto_plane_camera_space(
        laser_image_coord,
        lens_calibration.camera_matrix,
        lens_calibration.inverted_camera_matrix,
    )

    if np.any(np.isnan(laser_coord_3d)):
        return None

    return laser_coord_3d


@ray.remote(vram_mb=1536)
def evaluate(
    input_file: Path,
    laser_label_studio_json_path: Path | None,
    lens_calibration: LensCalibration,
    laser_calibration: LaserCalibration,
    rows: int,
    columns: int,
    square_size: float,
    debug_root: Path,
):
    debug_path = debug_root / "evaluation" / "laser"
    debug_path.mkdir(exist_ok=True, parents=True)

    png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")

    raw_processor = RawProcessor()

    try:
        image = uint16_2_uint8(raw_processor.process(input_file))
    except:
        return None

    image_rectifier = ImageRectifier(lens_calibration)
    image = image_rectifier.rectify(image)

    if laser_label_studio_json_path is not None:
        laser_detector = LabelStudioLaserDetector(
            input_file, laser_label_studio_json_path
        )
    else:
        raise NotImplementedError

    laser_image_coord = laser_detector.find_laser(image)
    if laser_image_coord is None:
        return None

    laser_detection_path = debug_path / f"detection_{png_name}"
    if laser_detection_path.exists():
        laser_detection_path.unlink()

    laser_detection = cv2.circle(
        image,
        np.round(laser_image_coord).astype(int),
        radius=5,
        color=(255, 0, 0),
        thickness=-1,
    )
    cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)

    checkerboard_detector = CheckerboardDetector(
        image, rows, columns, square_size * 10**-3
    )

    if not checkerboard_detector.is_valid():
        return None

    true_laser_coord_3d = checkerboard_detector.project_point_onto_plane_camera_space(
        laser_image_coord,
        lens_calibration.camera_matrix,
        lens_calibration.inverted_camera_matrix,
    )
    projected_depth_map = LaserDepthMap(
        laser_image_coord, lens_calibration, laser_calibration
    )

    return (true_laser_coord_3d[2] - projected_depth_map.depth_map[0]) ** 2


class CalibrateLaser(Command):
    @property
    def name(self) -> str:
        return "calibrate-laser"

    @property
    def description(self) -> str:
        return "Calibrates the laser for the FishSense Lite product line."

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
        "--rows",
        short_name="-r",
        required=True,
        help="The number of rows in the checkerboard.",
    )
    def rows(self) -> int:
        return self.__rows

    @rows.setter
    def rows(self, value: int):
        self.__rows = value

    @property
    @argument(
        "--columns",
        short_name="-c",
        required=True,
        help="The number of columns in the checkerboard.",
    )
    def columns(self) -> int:
        return self.__columns

    @columns.setter
    def columns(self, value: int):
        self.__columns = value

    @property
    @argument(
        "--square-size",
        short_name="-s",
        required=True,
        help="The size of a checkerboard square in mm.",
    )
    def square_size(self) -> float:
        return self.__square_size

    @square_size.setter
    def square_size(self, value: float):
        self.__square_size = value

    @property
    @argument(
        "--output",
        short_name="-o",
        required=True,
        help="The path to store the resulting calibration.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument("--overwrite", flag=True, help="Overwrite the calibration if it exists.")
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
        self.__laser_label_studio_json: str = None
        self.__rows: int = None
        self.__columns: int = None
        self.__square_size: float = None
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

        files = [Path(f) for g in self.data for f in glob(g)]
        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        futures = [
            execute.remote(
                f,
                laser_label_studio_json_path,
                lens_calibration,
                self.rows,
                self.columns,
                self.square_size,
                debug_path,
            )
            for f in files
        ]

        laser_points_3d = [
            p for p in self.tqdm(futures, total=len(files)) if p is not None
        ]
        laser_points_3d.sort(key=lambda x: x[2])
        laser_points_3d = np.array(laser_points_3d)

        laser_calibration = LaserCalibration()
        laser_calibration.plane_calibrate(laser_points_3d)

        futures = [
            evaluate.remote(
                f,
                laser_label_studio_json_path,
                lens_calibration,
                laser_calibration,
                self.rows,
                self.columns,
                self.square_size,
                debug_path,
            )
            for f in files
        ]

        square_errors = [
            p for p in self.tqdm(futures, total=len(files)) if p is not None
        ]
        error = np.sqrt(np.array(square_errors).mean())

        print(error)

        output_path = Path(self.output_path)

        if output_path.exists() and self.overwrite:
            output_path.unlink()

        laser_calibration.save(output_path)

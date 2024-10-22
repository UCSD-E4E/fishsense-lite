from glob import glob
from pathlib import Path
from typing import List

import cv2
import numpy as np
import ray
import torch
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.image.image_processors import RawProcessor
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector
from pyfishsensedev.plane_detector.checkerboard_detector import CheckerboardDetector
from tqdm import tqdm


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
    estimated_laser_calibration: LaserCalibration,
    rows: int,
    columns: int,
    square_size: float,
) -> np.ndarray | None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_path = Path(".debug") / "calibration" / "laser"
    debug_path.mkdir(exist_ok=True, parents=True)

    png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")

    raw_processor = RawProcessor(enable_histogram_equalization=False)

    try:
        image = uint16_2_uint8(raw_processor.load_and_process(input_file))
    except:
        return None

    image_rectifier = ImageRectifier(lens_calibration)
    image = image_rectifier.rectify(image)

    laser_detector = NNLaserDetector(
        lens_calibration, estimated_laser_calibration, device
    )
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
        color=(0, 255, 0),
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
        "--laser-position",
        short_name="-p",
        nargs=3,
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
        help="The laser axis unit vector inputed as x y z for the FishSense Lite product line.",
    )
    def laser_axis(self) -> List[float]:
        return self.__laser_axis

    @laser_axis.setter
    def laser_axis(self, value: List[float]):
        self.__laser_axis = value

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

    def __init__(self):
        super().__init__()

        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_position: List[int] = None
        self.__laser_axis: List[float] = None
        self.__rows: int = None
        self.__columns: int = None
        self.__square_size: float = None
        self.__output_path: str = None
        self.__overwrite: bool = None

    def __call__(self):
        self.init_ray()

        files = [Path(f) for g in self.data for f in glob(g)]
        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        estimated_laser_calibration = LaserCalibration(
            np.array(self.laser_axis), np.array(self.laser_position)
        )

        futures = [
            execute.remote(
                f,
                lens_calibration,
                estimated_laser_calibration,
                self.rows,
                self.columns,
                self.square_size,
            )
            for f in files
        ]

        laser_points_3d = [
            p for p in tqdm(to_iterator(futures), total=len(files)) if p is not None
        ]
        laser_points_3d.sort(key=lambda x: x[2])
        laser_points_3d = np.array(laser_points_3d)

        laser_calibration = LaserCalibration()
        laser_calibration.plane_calibrate(
            laser_points_3d, estimated_laser_calibration, use_gauss_newton=False
        )

        output_path = Path(self.output_path)

        if output_path.exists() and self.overwrite:
            output_path.unlink()

        laser_calibration.save(output_path)

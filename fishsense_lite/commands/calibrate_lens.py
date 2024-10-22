from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import ray
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image.image_processors import RawProcessor
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


@ray.remote(num_gpus=0.1)
def execute(
    input_file: Path, rows: int, columns: int, square_size: float
) -> Tuple[np.ndarray | None, np.ndarray | None, int | None, int | None] | None:
    square_size *= 10**-3

    raw_processor = RawProcessor()

    try:
        image = uint16_2_uint8(raw_processor.load_and_process(input_file))
    except:
        return None, None, None, None

    height, width, _ = image.shape
    checkerboard_detector = CheckerboardDetector(image, rows, columns, square_size)

    if not checkerboard_detector.is_valid():
        return None, None, width, height

    return (
        checkerboard_detector.points_body_space,
        checkerboard_detector.points_image_space,
        width,
        height,
    )


class CalibrateLens(Command):
    @property
    def name(self) -> str:
        return "calibrate-lens"

    @property
    def description(self) -> str:
        return "Calibrates the lens for the FishSense Lite product line."

    @property
    @argument("data", required=True, help="A glob that represents the data to process.")
    def data(self) -> List[str]:
        return self.__data

    @data.setter
    def data(self, value: List[str]):
        self.__data = value

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

    def __init__(self) -> None:
        super().__init__()

        self.__data: List[str] = None
        self.__rows: int = None
        self.__columns: int = None
        self.__square_size: float = None
        self.__output_path: str = None
        self.__overwrite: bool = None

    def __call__(self):
        self.init_ray()

        files = [Path(f) for g in self.data for f in glob(g)]

        futures = [
            execute.remote(f, self.rows, self.columns, self.square_size) for f in files
        ]

        points_body_space, points_image_space, width, height = list(
            zip(
                *(
                    (b, i, w, h)
                    for b, i, w, h in tqdm(to_iterator(futures), total=len(files))
                    if b is not None
                )
            )
        )

        lens_calibration = LensCalibration()
        lens_calibration.plane_calibrate(
            points_body_space, points_image_space, width[0], height[0]
        )

        output_path = Path(self.output_path)

        if output_path.exists() and self.overwrite:
            output_path.unlink()

        lens_calibration.save(output_path)

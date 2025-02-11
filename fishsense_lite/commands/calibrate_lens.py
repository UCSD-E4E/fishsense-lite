from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import fishsense_common.ray as ray
import numpy as np
from fishsense_common.pluggable_cli import Command, argument
from pyaqua3ddev.image.image_processors import RawProcessor
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.plane_detector.checkerboard_detector import CheckerboardDetector

from fishsense_lite.utils import uint16_2_uint8


@ray.remote(vram_mb=615)
def execute(
    input_file: Path,
    rows: int,
    columns: int,
    square_size: float,
    debug_root: Path,
) -> Tuple[np.ndarray | None, np.ndarray | None, int | None, int | None] | None:
    debug_path = debug_root / "calibration" / "lens"
    debug_path.mkdir(exist_ok=True, parents=True)

    png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")

    square_size *= 10**-3

    raw_processor = RawProcessor()

    try:
        image = uint16_2_uint8(raw_processor.process(input_file))
    except:
        return None, None, None, None

    height, width, _ = image.shape
    checkerboard_detector = CheckerboardDetector(image, rows, columns, square_size)

    if not checkerboard_detector.is_valid():
        return None, None, width, height

    checkerboard_detection_path = debug_path / f"checkerboard_{png_name}"
    if checkerboard_detection_path.exists():
        checkerboard_detection_path.unlink()

    cv2.drawChessboardCorners(
        image, (rows, columns), checkerboard_detector.points_image_space, True
    )
    cv2.imwrite(checkerboard_detection_path.absolute().as_posix(), image)

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

    @property
    @argument("--debug-path", help="Sets the debug path for storing debug images.")
    def debug_path(self) -> str:
        return self.__debug_path

    @debug_path.setter
    def debug_path(self, value: str):
        self.__debug_path = value

    def __init__(self) -> None:
        super().__init__()

        self.__data: List[str] = None
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

        files = [Path(f) for g in self.data for f in glob(g)]

        futures = [
            execute.remote(f, self.rows, self.columns, self.square_size, debug_path)
            for f in files
        ]

        points_body_space, points_image_space, width, height = list(
            zip(
                *(
                    (b, i, w, h)
                    for b, i, w, h in self.tqdm(futures, total=len(files))
                    if b is not None
                )
            )
        )

        mask = np.zeros((height[0], width[0], 3), dtype=np.uint8)
        for points in points_image_space:
            top_left = points[0]
            bottom_left = points[self.rows - 1]
            top_right = points[-self.rows]
            bottom_right = points[-1]

            # print(files[0], top_left, bottom_left, top_right, bottom_right)

            pts = np.array([top_left, bottom_left, bottom_right, top_right], np.int32)
            pts = pts.reshape((-1, 1, 2))
            curr_mask = np.zeros((height[0], width[0], 3), dtype=np.uint8)
            cv2.fillPoly(curr_mask, [pts], (1, 1, 1))

            mask += curr_mask

        import matplotlib.pyplot as plt

        plt.imshow(mask[:, :, 0])
        plt.colorbar()
        plt.show()

        lens_calibration = LensCalibration()
        error = lens_calibration.plane_calibrate(
            points_body_space, points_image_space, width[0], height[0]
        )

        print(error)

        output_path = Path(self.output_path)

        if output_path.exists() and self.overwrite:
            output_path.unlink()

        lens_calibration.save(output_path)

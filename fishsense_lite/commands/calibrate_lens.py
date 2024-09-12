from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import ray
from bom_common.pluggable_cli import Plugin
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image.image_processors import RawProcessor
from pyfishsensedev.plane_detector.checkerboard_detector import CheckerboardDetector
from tqdm import tqdm
from wakepy import keep


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


class CalibrateLens(Plugin):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)

        parser.add_argument(
            "data", nargs="+", help="A glob that represents the data to process."
        )

        parser.add_argument(
            "-r",
            "--rows",
            dest="rows",
            required=True,
            type=int,
            help="The number of rows in the checkerboard.",
        )

        parser.add_argument(
            "-c",
            "--columns",
            dest="columns",
            required=True,
            type=int,
            help="The number of columns in the checkerboard.",
        )

        parser.add_argument(
            "-s",
            "--square-size",
            dest="square_size",
            required=True,
            type=float,
            help="The size of a checkerboard square in mm.",
        )

        parser.add_argument(
            "-o",
            "--output",
            dest="output_path",
            required=True,
            help="The path to store the resulting calibration.",
        )

        parser.add_argument(
            "--overwrite",
            dest="overwrite",
            action="store_true",
            help="The path to store the resulting calibration.",
        )

    def __call__(self, args: Namespace):
        with keep.running():
            ray.init()

            files = [Path(f) for g in args.data for f in glob(g)]

            futures = [
                execute.remote(f, args.rows, args.columns, args.square_size)
                for f in files
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

            output_path = Path(args.output_path)

            if output_path.exists() and args.overwrite:
                output_path.unlink()

            lens_calibration.save(output_path)

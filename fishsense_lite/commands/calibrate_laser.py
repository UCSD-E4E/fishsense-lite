from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path

import numpy as np
import torch
from bom_common.pluggable_cli import Plugin
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.image.image_processors import RawProcessor
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector
from tqdm import tqdm
from wakepy import keep


def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    estimated_laser_calibration: LaserCalibration,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_processor_hist_eq = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    try:
        image = raw_processor_hist_eq.load_and_process(input_file)
        image_dark = raw_processor.load_and_process(input_file)
    except:
        return None

    image_rectifier = ImageRectifier(lens_calibration)
    image = image_rectifier.rectify(image)
    image_dark = image_rectifier.rectify(image_dark)

    laser_detector = NNLaserDetector(
        lens_calibration, estimated_laser_calibration, device
    )
    laser_image_coordinates = laser_detector.find_laser(image_dark)

    if laser_image_coordinates is None:
        return None


class CalibrateLaser(Plugin):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)

        parser.add_argument(
            "data", nargs="+", help="A glob that represents the data to process."
        )

        parser.add_argument(
            "-c",
            "--lens-calibration",
            dest="lens_calibration",
            required=True,
            help="Lens calibration package for the FishSense Lite.",
        )

        parser.add_argument(
            "-p",
            "--laser-position",
            nargs="+",
            dest="laser_position",
            type=int,
            required=True,
            help="The laser position in centimeter inputed as x y z for the FishSense Lite product line.",
        )

        parser.add_argument(
            "-a",
            "--laser-axis",
            nargs="+",
            dest="laser_axis",
            type=float,
            required=True,
            help="The laser axis unit vector inputed as x y z for the FishSense Lite product line.",
        )

    def __call__(self, args: Namespace):
        with keep.running():
            files = [Path(f) for g in args.data for f in glob(g)]
            lens_calibration = LensCalibration()
            lens_calibration.load(Path(args.lens_calibration))

            estimated_laser_calibration = LaserCalibration(
                np.array(args.laser_axis), np.array(args.laser_position)
            )

            list(
                tqdm(
                    (
                        execute(f, lens_calibration, estimated_laser_calibration)
                        for f in files
                    ),
                    total=len(files),
                )
            )

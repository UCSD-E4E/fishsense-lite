from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import ray
import torch
from bom_common.pluggable_cli import Plugin
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.image.image_processors import RawProcessor
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector
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
    pdf: Pdf,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_path = Path("~/.fishsense-lite") / "debug" / "field-calibration" / "laser"
    debug_path.mkdir(exist_ok=True, parents=True)

    png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")

    raw_processor_histogram_equalization = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    image = uint16_2_uint8(
        raw_processor_histogram_equalization.load_and_process(input_file)
    )
    image_dark = uint16_2_uint8(raw_processor.load_and_process(input_file))

    image_rectifier = ImageRectifier(lens_calibration)

    image = image_rectifier.rectify(image)
    image_dark = image_rectifier.rectify(image_dark)

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
        image_dark,
        np.round(laser_image_coord).astype(int),
        radius=5,
        color=(0, 255, 0),
        thickness=-1,
    )
    cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)


class FieldCalibrateLaser(Plugin):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)

        parser.add_argument(
            "data", nargs="+", help="A glob that represents the data to process."
        )

        parser.add_argument(
            "-l",
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

        parser.add_argument(
            "--pdf",
            dest="pdf",
            required=True,
            help="The PDF scan of a dive slate configured to be used for the FishSense Lite product line.",
        )

    def __call__(self, args: Namespace):
        files = [Path(f) for g in args.data for f in glob(g)]
        lens_calibration = LensCalibration()
        lens_calibration.load(Path(args.lens_calibration))

        estimated_laser_calibration = LaserCalibration(
            np.array(args.laser_axis), np.array(args.laser_position)
        )

        pdf = Pdf(Path(args.pdf))

        # list(
        #     tqdm(
        #         (
        #             execute(f, lens_calibration, estimated_laser_calibration, pdf)
        #             for f in files
        #         ),
        #         total=len(files),
        #     )
        # )

        futures = [
            execute.remote(f, lens_calibration, estimated_laser_calibration, pdf)
            for f in files
        ]

        laser_points_3d = [
            p for p in tqdm(to_iterator(futures), total=len(files)) if p is not None
        ]

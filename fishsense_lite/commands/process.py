from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path

import numpy as np
import ray
import torch
from bom_common.pluggable_cli import Plugin
from pyfishsense import (
    FishHeadTailDetector,
    FishSegmentationInference,
    ImageRectifier,
    LaserDetector,
    RawProcessor,
    WorldPointHandler,
)
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
    input_file: Path, lens_calibration_path: Path, laser_calibration_path: Path
) -> float | None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_processor_hist_eq = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    image_rectifier = ImageRectifier(lens_calibration_path)

    img = raw_processor_hist_eq.load_and_process(input_file)
    img_dark = raw_processor.load_and_process(input_file)

    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = uint16_2_uint8(img)
    img_dark8 = uint16_2_uint8(img_dark)

    laser_detector = LaserDetector(
        lens_calibration_path, laser_calibration_path, device
    )
    laser_coords = laser_detector.find_laser(img_dark8)

    if laser_coords is None:
        return input_file, None

    fish_segmentation_inference = FishSegmentationInference(device)
    segmentations = fish_segmentation_inference.inference(img8)

    if segmentations.sum() == 0:
        return input_file, None

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[laser_coords[1], laser_coords[0]]] = True

    if mask.sum() == 0:
        return input_file, None

    fish_head_tail_detector = FishHeadTailDetector()
    left_coord, right_coord = fish_head_tail_detector.find_head_tail(mask)

    world_point_handler = WorldPointHandler(
        lens_calibration_path, laser_calibration_path
    )
    laser_coords3d = world_point_handler.calculate_laser_parallax(laser_coords)

    left_coord3d, right_coord3d = (
        world_point_handler.calculate_world_coordinates_with_depth(
            left_coord, right_coord, laser_coords3d[2]
        )
    )
    length = np.linalg.norm(left_coord3d - right_coord3d)

    return input_file, length


class Process(Plugin):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)

        parser.add_argument(
            "data", nargs="+", help="A glob that represents the data to process."
        )

        parser.add_argument(
            "-c", "--lens-calibration", dest="lens_calibration", required=True
        )

        parser.add_argument(
            "-l", "--laser-calibration", dest="laser_calibration", required=True
        )

    def __call__(self, args: Namespace):
        files = {Path(f) for g in args.data for f in glob(g, recursive=True)}
        lens_calibration_path = Path(args.lens_calibration)
        laser_calibration_path = Path(args.laser_calibration)

        futures = [
            execute.remote(f, lens_calibration_path, laser_calibration_path)
            for f in files
        ]

        files, lengths = list(zip(*tqdm(to_iterator(futures), total=len(files))))
        print(lengths)

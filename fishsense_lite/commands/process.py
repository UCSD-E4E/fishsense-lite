from argparse import ArgumentParser, Namespace
from glob import glob
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import ray
import torch
from bom_common.pluggable_cli import Plugin
from pyfishsense import sum_as_string
from pyfishsensedev import (
    FishHeadTailDetector,
    FishSegmentationInference,
    ImageRectifier,
    LaserDetector,
    RawProcessor,
    WorldPointHandler,
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


@ray.remote(num_gpus=0.1)
def execute_orf(
    input_file: Path, lens_calibration_path: Path, laser_calibration_path: Path
) -> Tuple[Path, ResultStatus, float]:
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
        return input_file, ResultStatus.FAILED_LASER_COORDS, None

    fish_segmentation_inference = FishSegmentationInference(device)
    segmentations = fish_segmentation_inference.inference(img8)

    if segmentations.sum() == 0:
        return input_file, ResultStatus.FAILED_SEGMENTATION, None

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[laser_coords[1], laser_coords[0]]] = True

    if mask.sum() == 0:
        return input_file, ResultStatus.FAILED_LASER_SEGMENTATION_INTERSECTION, None

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

    return input_file, ResultStatus.SUCCESS, length


def _execute_csv_python_tool(
    csv_path: Path, df: pd.DataFrame, world_point_handler: WorldPointHandler
):
    for index, row in df.iterrows():
        # name	laser.x	laser.y	head.x	head.y	tail.x	tail.y
        file_name = Path(row["name"]).parts[-1]

        head_coords = np.array([row["head.x"], row["head.y"]])
        tail_coords = np.array([row["tail.x"], row["tail.y"]])
        laser_coords = np.array([row["laser.x"], row["laser.y"]])

        laser_coords3d = world_point_handler.calculate_laser_parallax(laser_coords)

        left_coord3d, right_coord3d = (
            world_point_handler.calculate_world_coordinates_with_depth(
                head_coords, tail_coords, laser_coords3d[2]
            )
        )
        length = np.linalg.norm(left_coord3d - right_coord3d)

        yield csv_path.parent / file_name, ResultStatus.SUCCESS, length


def _execute_csv_viame(
    csv_path: Path, lines: List[str], world_point_handler: WorldPointHandler
):
    for line in lines[2:]:
        parts = [p.strip() for p in line.split(",") if p.strip()]
        file_name = parts[1]

        is_head_tail = parts[-2].startswith("(kp) head") and parts[-1].startswith(
            "(kp) tail"
        )
        is_laser = parts[-1].startswith("(kp) head")

        if is_head_tail:
            # (kp) head 1632 1497
            head_coords = np.array([int(p) for p in parts[-2].split(" ")[-2:]])
            # (kp) tail 2340 1186
            tail_coords = np.array([int(p) for p in parts[-1].split(" ")[-2:]])
        elif is_laser:
            if prev_file_name != file_name:
                continue

            # (kp) head 1632 1497
            laser_coords = np.array([int(p) for p in parts[-1].split(" ")[-2:]])
            laser_coords3d = world_point_handler.calculate_laser_parallax(laser_coords)

            left_coord3d, right_coord3d = (
                world_point_handler.calculate_world_coordinates_with_depth(
                    head_coords, tail_coords, laser_coords3d[2]
                )
            )
            length = np.linalg.norm(left_coord3d - right_coord3d)

            yield csv_path.parent / file_name, ResultStatus.SUCCESS, length
        else:
            continue

        prev_file_name = file_name


def execute_csv(
    input_files: Set[Path], lens_calibration_path: Path, laser_calibration_path: Path
):
    world_point_handler = WorldPointHandler(
        lens_calibration_path.absolute(), laser_calibration_path.absolute()
    )

    for file in input_files:
        with file.open("r") as f:
            lines = f.readlines()

        is_viame = lines[0].startswith("# 1: Detection or Track-id")
        is_python_tool = lines[0].startswith("name")

        if is_viame:
            for result in _execute_csv_viame(file, lines, world_point_handler):
                yield result
        elif is_python_tool:
            df = pd.read_csv(file.as_posix())
            for result in _execute_csv_python_tool(file, df, world_point_handler):
                yield result


class Process(Plugin):
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
            "-l",
            "--laser-calibration",
            dest="laser_calibration",
            required=True,
            help="Laser calibration package for the FishSense Lite.",
        )

        parser.add_argument(
            "-o",
            "--output",
            required=True,
            help="The output file containing length data.",
        )

        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite images previously computed.",
        )

    def __call__(self, args: Namespace):
        with Database(Path(args.output)) as database:
            print(f"sum_as_string(1, 2) in Rust: {sum_as_string(1, 2)}")

            files = {Path(f) for g in args.data for f in glob(g, recursive=True)}

            if not args.overwrite:
                prev_files = database.get_files()
                files.difference_update(prev_files)

            orf_files = {f for f in files if f.suffix.lower() == ".orf"}
            csv_files = {f for f in files if f.suffix.lower() == ".csv"}

            lens_calibration_path = Path(args.lens_calibration)
            laser_calibration_path = Path(args.laser_calibration)

            orf_futures = [
                execute_orf.remote(f, lens_calibration_path, laser_calibration_path)
                for f in orf_files
            ]

            results: Iterable[Tuple[Path, ResultStatus, float]] = tqdm(
                chain(
                    to_iterator(orf_futures),
                    execute_csv(
                        csv_files, lens_calibration_path, laser_calibration_path
                    ),
                ),
                total=len(orf_files) + len(csv_files),
            )

            # results = tqdm(
            #     (
            #         execute(f, lens_calibration_path, laser_calibration_path)
            #         for f in files
            #     ),
            #     total=len(files),
            # )

            for file, result_status, length in results:
                database.insert_data(file, result_status, length)

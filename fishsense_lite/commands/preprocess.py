from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path
from typing import Iterable

import cv2
import ray
from bom_common.pluggable_cli import Plugin
from pyfishsensedev.image import ImageRectifier, RawProcessor
from tqdm import tqdm


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@ray.remote(num_gpus=0.1)
def execute(input_file: Path, lens_calibration_path: Path):
    raw_processor_hist_eq = RawProcessor()
    image_rectifier = ImageRectifier(lens_calibration_path)

    img = raw_processor_hist_eq.load_and_process(input_file)
    img = image_rectifier.rectify(img)

    output_file = Path(input_file.as_posix().replace(input_file.suffix, ".png"))

    cv2.imwrite(output_file.absolute().as_posix(), img)


class Preprocess(Plugin):
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
            "--overwrite",
            action="store_true",
            help="Overwrite images previously computed.",
        )

    def __call__(self, args: Namespace):
        ray.init()

        files = {Path(f) for g in args.data for f in glob(g, recursive=True)}
        lens_calibration_path = Path(args.lens_calibration)

        futures = [execute.remote(f, lens_calibration_path) for f in files]

        # Hack to force processing
        _ = list(tqdm(to_iterator(futures), total=len(files)))

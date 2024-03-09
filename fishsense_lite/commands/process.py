from argparse import ArgumentParser, Namespace
from glob import glob
from pathlib import Path

import ray
from bom_common.pluggable_cli import Plugin
from pyfishsense import RawProcessor
from tqdm import tqdm


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@ray.remote
def execute(input_file: Path) -> float | None:
    raw_processor_hist_eq = RawProcessor()
    raw_processor = RawProcessor(enable_histogram_equalization=False)

    # image_rectifier = ImageRectifier(lens_calibration_path)

    img = raw_processor_hist_eq.load_and_process(input_file)
    img_dark = raw_processor.load_and_process(input_file)

    # img = image_rectifier.rectify(img)
    # img_dark = image_rectifier.rectify(img_dark)

    # img8 = uint16_2_uint8(img)
    # img_dark8 = uint16_2_uint8(img_dark)


class Process(Plugin):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)

        parser.add_argument(
            "data", nargs="+", help="A glob that represents the data to process."
        )

    def __call__(self, args: Namespace):
        files = {Path(f) for g in args.data for f in glob(g, recursive=True)}
        futures = [execute.remote(f) for f in files]

        list(tqdm(to_iterator(futures), total=len(files)))

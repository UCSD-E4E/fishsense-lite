from glob import glob
from pathlib import Path
from typing import List

import cv2
import ray
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image import ImageRectifier, RawProcessor
from tqdm import tqdm


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


@ray.remote(num_gpus=0.1)
def execute(
    input_file: Path,
    disable_histogram_equalization: bool,
    lens_calibration: LensCalibration,
):
    raw_processor_hist_eq = RawProcessor(
        enable_histogram_equalization=not disable_histogram_equalization
    )
    image_rectifier = ImageRectifier(lens_calibration)

    img = raw_processor_hist_eq.load_and_process(input_file)
    img = image_rectifier.rectify(img)

    output_file = Path(input_file.as_posix().replace(input_file.suffix, ".png"))

    cv2.imwrite(output_file.absolute().as_posix(), img)


class Preprocess(Command):
    @property
    def name(self) -> str:
        return "preprocess"

    @property
    def description(self) -> str:
        return "Preprocess data from the FishSense Lite product line."

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
        "--disable-histogram-equalization",
        flag=True,
        help="Disables histogram equalization when processing images.",
    )
    def disable_histogram_equalization(self) -> bool:
        return self.__disable_histogram_equalization

    @disable_histogram_equalization.setter
    def disable_histogram_equalization(self, value: bool):
        self.__disable_histogram_equalization = value

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
        self.__disable_histogram_equalization: bool = None
        self.__overwrite: bool = None

    def __call__(self):
        ray.init()

        files = {Path(f) for g in self.data for f in glob(g, recursive=True)}
        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        futures = [
            execute.remote(f, self.disable_histogram_equalization, lens_calibration)
            for f in files
        ]

        # Hack to force processing
        _ = list(tqdm(to_iterator(futures), total=len(files)))

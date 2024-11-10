from glob import glob
from pathlib import Path
from typing import List

import cv2
import fishsense_common.ray as ray
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image import ImageRectifier, RawProcessor

from fishsense_lite.utils import get_output_file, get_root, uint16_2_uint8


@ray.remote(vram_mb=615)
def execute(
    input_file: Path,
    disable_histogram_equalization: bool,
    lens_calibration: LensCalibration,
    root: Path,
    output: Path,
    format: str,
    overwrite: bool,
):
    output_file = get_output_file(input_file, root, output, format)

    if not overwrite and output_file.exists():
        return

    raw_processor_hist_eq = RawProcessor(
        enable_histogram_equalization=not disable_histogram_equalization
    )
    image_rectifier = ImageRectifier(lens_calibration)

    try:
        img = raw_processor_hist_eq.load_and_process(input_file)
    except:
        return
    img = image_rectifier.rectify(img)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "jpg":
        img = uint16_2_uint8(img)

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
    @argument(
        "--output",
        short_name="-o",
        required=True,
        help="The path to store the resulting database.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument(
        "--format",
        short_name="-f",
        default="png",
        help="The image format to save the preprocessed ata in.  PNG is saved as 16 bit.  JPG is saved as 8 bit.",
    )
    def format(self) -> str:
        return self.__format

    @format.setter
    def format(self, value: str):
        self.__format = value

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
        self.__output_path: str = None
        self.__format: str = None
        self.__overwrite: bool = None

    def __call__(self):
        self.init_ray()

        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}

        # Find the singular path that defines the root of all of our data.
        root = get_root(files)

        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        output = Path(self.output_path)

        futures = [
            execute.remote(
                f,
                self.disable_histogram_equalization,
                lens_calibration,
                root,
                output,
                self.format,
                self.overwrite,
            )
            for f in files
        ]

        # Hack to force processing
        _ = list(self.tqdm(futures, total=len(files)))

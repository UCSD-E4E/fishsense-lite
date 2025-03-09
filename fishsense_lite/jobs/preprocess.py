from glob import glob
from pathlib import Path
from typing import Any, Iterable, List

import cv2
from fishsense_common.scheduling.arguments import argument
from fishsense_common.scheduling.job_definition import JobDefinition
from fishsense_common.scheduling.ray_job import RayJob
from pyaqua3ddev.image.image_processors import RawProcessor
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image import ImageRectifier
from skimage.util import img_as_ubyte

from fishsense_lite.utils import get_output_file, get_root


def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    root: Path,
    output: Path,
    format: str,
    overwrite: bool,
):
    output_file = get_output_file(input_file, root, output, format)

    if not overwrite and output_file.exists():
        return

    raw_processor = RawProcessor()
    image_rectifier = ImageRectifier(lens_calibration)

    try:
        img = raw_processor.process(input_file)
    except:
        return
    img = image_rectifier.rectify(img)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "jpg":
        img = img_as_ubyte(img)

    output_file.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(output_file.absolute().as_posix(), img)


class Preprocess(RayJob):
    name = "preprocess"

    @property
    def job_count(self) -> int:
        return len({f for g in self.data for f in glob(g, recursive=True)})

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
        "lens-calibration",
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
        "output",
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
        "format",
        default="png",
        help="The image format to save the preprocessed ata in.  PNG is saved as 16 bit.  JPG is saved as 8 bit.",
    )
    def format(self) -> str:
        return self.__format

    @format.setter
    def format(self, value: str):
        self.__format = value

    @property
    @argument("overwrite", help="Overwrite the calibration if it exists.")
    def overwrite(self) -> bool:
        return self.__overwrite

    @overwrite.setter
    def overwrite(self, value: bool):
        self.__overwrite = value

    def __init__(self, job_defintion: JobDefinition):
        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__output_path: str = None
        self.__format: str = None
        self.__overwrite: bool = None

        super().__init__(job_defintion, execute, vram_mb=615)

    def prologue(self) -> Iterable[Iterable[Any]]:
        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}

        # Find the singular path that defines the root of all of our data.
        root = get_root(files)

        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        output = Path(self.output_path)

        return (
            (
                f,
                lens_calibration,
                root,
                output,
                self.format,
                self.overwrite,
            )
            for f in files
        )

    def epiloge(self, results: Iterable[Any]):
        # Hack to force processing
        _ = list(results)

from glob import glob
from pathlib import Path
from typing import Any, Iterable, List

from fishsense_common.pipeline.pipeline import Pipeline
from fishsense_common.scheduling.arguments import argument
from fishsense_common.scheduling.job_definition import JobDefinition
from fishsense_common.scheduling.ray_job import RayJob
from fishsense_common.utils.cuda import set_opencv_opencl_device
from pyfishsensedev.calibration import LensCalibration
from upath import UPath

from fishsense_lite.pipeline.tasks.image_rectifier import image_rectifier
from fishsense_lite.pipeline.tasks.process_raw import process_raw
from fishsense_lite.pipeline.tasks.save_output import save_output
from fishsense_lite.utils import get_root


def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    root: Path,
    output: Path,
    format: str,
):
    set_opencv_opencl_device()
    pipeline = Pipeline(process_raw, image_rectifier, save_output)

    pipeline(
        input_file=input_file,
        lens_calibration=lens_calibration,
        root=root,
        output=output,
        format=format,
    )


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

    def __init__(
        self,
        job_defintion: JobDefinition,
        input_filesystem: Any,
        output_filesystem: Any,
    ):
        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__output_path: str = None
        self.__format: str = None

        super().__init__(
            job_defintion, input_filesystem, output_filesystem, execute, vram_mb=615
        )

    def prologue(self) -> Iterable[Iterable[Any]]:
        files = {
            UPath(f).absolute()
            for g in self.data
            for f in self.input_filesystem.glob(g, recursive=True)
        }

        # Find the singular path that defines the root of all of our data.
        root = get_root(files)

        lens_calibration = LensCalibration()
        lens_calibration.load(UPath(self.lens_calibration))

        output = UPath(self.output_path, **self.output_filesystem.storage_options)

        return (
            (
                f,
                lens_calibration,
                root,
                output,
                self.format,
            )
            for f in files
        )

    def epilogue(self, results: Iterable[Any]):
        # Hack to force processing
        _ = list(results)

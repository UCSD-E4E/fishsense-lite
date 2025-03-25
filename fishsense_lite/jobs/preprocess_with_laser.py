from glob import glob
from pathlib import Path
from typing import Any, Iterable, List

from fishsense_common.pipeline.decorators import task
from fishsense_common.pipeline.pipeline import Pipeline
from fishsense_common.scheduling.arguments import argument
from fishsense_common.scheduling.job_definition import JobDefinition
from fishsense_common.scheduling.ray_job import RayJob
from pyaqua3ddev.laser.single_laser.label_studio_laser_detector import (
    LabelStudioLaserDetector,
)
from pyaqua3ddev.laser.single_laser.laser_detector import LaserDetector
from pyaqua3ddev.laser.single_laser.psql_laser_detector import PSqlLabelDetector
from pyfishsensedev.calibration import LensCalibration

from fishsense_lite.pipeline.tasks.detect_laser import detect_laser
from fishsense_lite.pipeline.tasks.display_laser import display_laser
from fishsense_lite.pipeline.tasks.image_rectifier import image_rectifier
from fishsense_lite.pipeline.tasks.process_raw import process_raw
from fishsense_lite.pipeline.tasks.save_output import save_output
from fishsense_lite.utils import (
    PSqlConnectionString,
    get_root,
    parse_psql_connection_string,
)


def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    laser_labels_path: Path,
    connection_string: PSqlConnectionString,
    root: Path,
    output: Path,
    format: str,
):
    laser_detector: LaserDetector = None
    if laser_labels_path is not None:
        laser_detector = LabelStudioLaserDetector(input_file, laser_labels_path)
    elif connection_string is not None:
        laser_detector = PSqlLabelDetector(
            input_file,
            connection_string.dbname,
            connection_string.username,
            connection_string.password,
            connection_string.host,
            port=connection_string.port,
        )
    else:
        raise NotImplementedError

    pipeline = Pipeline(
        process_raw,
        image_rectifier,
        detect_laser,
        task("img")(display_laser),
        save_output,
    )

    pipeline(
        input_file=input_file,
        lens_calibration=lens_calibration,
        laser_detector=laser_detector,
        root=root,
        output=output,
        format=format,
    )


class PreprocessWithLaser(RayJob):
    name = "preprocess_with_laser"

    @property
    def job_count(self) -> int:
        return len({f for g in self.data for f in glob(g, recursive=True)})

    @property
    def description(self) -> str:
        return "Preprocess data from the FishSense Lite product line and include laser label."

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
    @argument(
        "laser-labels",
        help="The path to the laser labels export from Label Studio.",
    )
    def laser_labels(self) -> str:
        return self.__laser_labels

    @laser_labels.setter
    def laser_labels(self, value: str):
        self.__laser_labels = value

    @property
    @argument(
        "psql-connection-string",
        help="The connection string to the Postgres database.",
    )
    def psql_connection_string(self) -> str:
        return self.__psql_connection_string

    @psql_connection_string.setter
    def psql_connection_string(self, value: str):
        self.__psql_connection_string = value

    def __init__(self, job_defintion: JobDefinition):
        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__output_path: str = None
        self.__format: str = None
        self.__laser_labels: str = None
        self.__psql_connection_string: str = None

        super().__init__(job_defintion, execute, vram_mb=615)

    def prologue(self) -> Iterable[Iterable[Any]]:
        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}

        # Find the singular path that defines the root of all of our data.
        root = get_root(files)

        laser_labels_path = (
            Path(self.laser_labels) if self.laser_labels is not None else None
        )

        psql_connection_string = parse_psql_connection_string(
            self.psql_connection_string
        )

        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        output = Path(self.output_path)

        return (
            (
                f,
                lens_calibration,
                laser_labels_path,
                psql_connection_string,
                root,
                output,
                self.format,
            )
            for f in files
        )

    def epiloge(self, results: Iterable[Any]):
        # Hack to force processing
        _ = list(results)

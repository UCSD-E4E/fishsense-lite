from glob import glob
from pathlib import Path
from typing import Any, Iterable, List, Tuple

from fishsense_common.pipeline.pipeline import Pipeline
from fishsense_common.scheduling.arguments import argument
from fishsense_common.scheduling.job_definition import JobDefinition
from fishsense_common.scheduling.ray_job import RayJob
from fishsense_common.utils.cuda import set_opencv_opencl_device
from pyfishsensedev.calibration import LaserCalibration, LensCalibration

from fishsense_lite.database import Database
from fishsense_lite.pipeline.tasks.calculate_length import calculate_length
from fishsense_lite.pipeline.tasks.calculate_points_of_interest import (
    calculate_points_of_interest,
)
from fishsense_lite.pipeline.tasks.calculate_segmentation_mask import (
    calculate_segmentation_mask,
)
from fishsense_lite.pipeline.tasks.detect_laser import detect_laser
from fishsense_lite.pipeline.tasks.get_laser_detector import get_laser_detector
from fishsense_lite.pipeline.tasks.get_points_of_interest_detector import (
    get_points_of_interest_detector,
)
from fishsense_lite.pipeline.tasks.get_segmentation_model import get_segmentation_model
from fishsense_lite.pipeline.tasks.image_rectifier import image_rectifier
from fishsense_lite.pipeline.tasks.make_debug_path import make_debug_path
from fishsense_lite.pipeline.tasks.process_raw import process_raw
from fishsense_lite.utils import PSqlConnectionString, parse_psql_connection_string


def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    laser_calibration: LaserCalibration,
    laser_labels_path: Path,
    head_tail_labels_path: Path,
    connection_string: PSqlConnectionString,
    use_sql_for_laser_labels: bool,
    use_sql_for_head_tail_labels: bool,
    debug_root: Path,
) -> Tuple[Path, str, float]:
    set_opencv_opencl_device()
    pipeline = Pipeline(
        make_debug_path,
        process_raw,
        image_rectifier,
        get_laser_detector,
        detect_laser,
        get_segmentation_model,
        calculate_segmentation_mask,
        get_points_of_interest_detector,
        calculate_points_of_interest,
        calculate_length,
    )

    statuses, result = pipeline(
        input_file=input_file,
        lens_calibration=lens_calibration,
        laser_calibration=laser_calibration,
        laser_labels_path=laser_labels_path,
        head_tail_labels_path=head_tail_labels_path,
        use_sql_for_laser_labels=use_sql_for_laser_labels,
        use_sql_for_head_tail_labels=use_sql_for_head_tail_labels,
        connection_string=connection_string,
        debug_root=debug_root,
    )

    if any(s for s in statuses):
        return input_file, "SUCCESS", result
    else:
        return input_file, [k for k, v in statuses.items() if not v][0], result


class Process(RayJob):
    name = "process"

    @property
    def job_count(self) -> int:
        return len({f for g in self.data for f in glob(g, recursive=True)})

    @property
    def description(self) -> str:
        return "Process data from the FishSense Lite product line."

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
        "laser-calibration",
        required=True,
        help="Laser calibration package for the FishSense Lite.",
    )
    def laser_calibration(self) -> str:
        return self.__laser_calibration

    @laser_calibration.setter
    def laser_calibration(self, value: str):
        self.__laser_calibration = value

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
        "head-tail-labels",
        help="The path to the head tail labels export from Label Studio.",
    )
    def head_tail_labels(self) -> str:
        return self.__head_tail_labels

    @head_tail_labels.setter
    def head_tail_labels(self, value: str):
        self.__head_tail_labels = value

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

    @property
    @argument(
        "use-sql-for-laser-labels",
        default=False,
        help="Use SQL for laser labels. This takes precedence over the laser labels path.",
    )
    def use_sql_for_laser_labels(self) -> bool:
        return self.__use_sql_for_laser_labels

    @use_sql_for_laser_labels.setter
    def use_sql_for_laser_labels(self, value: bool):
        self.__use_sql_for_laser_labels = value

    @property
    @argument(
        "use-sql-for-head-tail-labels",
        default=False,
        help="Use SQL for head tail labels. This takes precedence over the head tail labels path.",
    )
    def use_sql_for_head_tail_labels(self) -> bool:
        return self.__use_sql_for_head_tail_labels

    @use_sql_for_head_tail_labels.setter
    def use_sql_for_head_tail_labels(self, value: bool):
        self.__use_sql_for_head_tail_labels = value

    @property
    @argument(
        "output",
        required=True,
        help="The path to store the resulting calibration.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument("debug-path", help="Sets the debug path for storing debug images.")
    def debug_path(self) -> str:
        return self.__debug_path

    @debug_path.setter
    def debug_path(self, value: str):
        self.__debug_path = value

    def __init__(self, job_defintion: JobDefinition):
        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_labels: str = None
        self.__head_tail_labels: str = None
        self.__psql_connection_string: str = None
        self.__use_sql_for_laser_labels: bool = False
        self.__use_sql_for_head_tail_labels: bool = False
        self.__output_path: str = None
        self.__debug_path: str = None

        super().__init__(job_defintion, execute, vram_mb=1536)

    def prologue(self) -> Iterable[Iterable[Any]]:
        if self.debug_path is None:
            self.debug_path = ".debug"

        debug_path = Path(self.debug_path)

        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}

        lens_calibration_path = Path(self.lens_calibration)
        laser_calibration_path = Path(self.laser_calibration)

        if not lens_calibration_path.exists() or not laser_calibration_path.exists():
            return ()

        lens_calibration = LensCalibration()
        lens_calibration.load(lens_calibration_path)

        laser_calibration = LaserCalibration()
        laser_calibration.load(laser_calibration_path)

        laser_labels_path = Path(self.laser_labels) if self.laser_labels else None
        head_tail_labels_path = (
            Path(self.head_tail_labels) if self.head_tail_labels else None
        )

        psql_connection_string = parse_psql_connection_string(
            self.psql_connection_string
        )

        return (
            (
                f,
                lens_calibration,
                laser_calibration,
                laser_labels_path,
                head_tail_labels_path,
                psql_connection_string,
                self.use_sql_for_laser_labels,
                self.use_sql_for_head_tail_labels,
                debug_path,
            )
            for f in files
        )

    def epilogue(self, results: Iterable[Tuple[Path, str, float]]) -> None:
        with Database(Path(self.output_path)) as database:
            for idx, (file, result_status, length) in enumerate(results):
                print(idx, file, result_status, length)

                database.insert_data(file, result_status, length)

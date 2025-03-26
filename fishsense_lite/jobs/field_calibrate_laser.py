from glob import glob
from pathlib import Path
from typing import Iterable, List

import numpy as np
from fishsense_common.pipeline.pipeline import Pipeline
from fishsense_common.scheduling.arguments import argument
from fishsense_common.scheduling.ray_job import RayJob
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.image.pdf import Pdf

from fishsense_lite.pipeline.tasks.calculate_laser_coord_3d_from_slate import (
    calculate_laser_coord_3d_from_slate,
)
from fishsense_lite.pipeline.tasks.detect_laser import detect_laser
from fishsense_lite.pipeline.tasks.get_laser_detector import get_laser_detector
from fishsense_lite.pipeline.tasks.image_rectifier import image_rectifier
from fishsense_lite.pipeline.tasks.make_debug_path import make_debug_path
from fishsense_lite.pipeline.tasks.process_raw import process_raw
from fishsense_lite.utils import PSqlConnectionString, parse_psql_connection_string


def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    laser_labels_path: Path,
    connection_string: PSqlConnectionString,
    pdf: Pdf,
    debug_root: Path,
) -> np.ndarray[float]:
    pipeline = Pipeline(
        make_debug_path,
        process_raw,
        image_rectifier,
        get_laser_detector,
        detect_laser,
        calculate_laser_coord_3d_from_slate,
        return_name="laser_coord_3d",
    )

    return pipeline(
        input_file=input_file,
        lens_calibration=lens_calibration,
        laser_labels_path=laser_labels_path,
        connection_string=connection_string,
        pdf=pdf,
        debug_root=debug_root,
    )


class FieldCalibrateLaser(RayJob):
    name = "field_calibrate_laser"

    @property
    def job_count(self) -> int:
        return len({f for g in self.data for f in glob(g, recursive=True)})

    @property
    def description(self) -> str:
        return "Perform the field calibration method."

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

    @property
    @argument(
        "pdf",
        required=True,
        help="The path to the PDF file.",
    )
    def pdf(self) -> str:
        return self.__pdf

    @pdf.setter
    def pdf(self, value: str):
        self.__pdf = value

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

    def __init__(self, job_definition, vram_mb=1536):
        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_labels: str = None
        self.__psql_connection_string: str = None
        self.__pdf: str = None
        self.__output_path: str = None
        self.__debug_path: str = None

        super().__init__(job_definition, execute, vram_mb)

    def prologue(self):
        if self.debug_path is None:
            self.debug_path = ".debug"

        debug_path = Path(self.debug_path)

        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}
        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        laser_labels_path = (
            Path(self.laser_labels) if self.laser_labels is not None else None
        )

        psql_connection_string = parse_psql_connection_string(
            self.psql_connection_string
        )

        pdf = Pdf(Path(self.pdf))

        return (
            (
                f,
                lens_calibration,
                laser_labels_path,
                psql_connection_string,
                pdf,
                debug_path,
            )
            for f in files
        )

    def epilogue(self, results: Iterable[np.ndarray[float]]):
        laser_points_3d = [p for p in results if p is not None]
        laser_points_3d.sort(key=lambda x: x[2])
        laser_points_3d = np.array(laser_points_3d)

        laser_calibration = LaserCalibration()
        laser_calibration.plane_calibrate(laser_points_3d)

        output_path = Path(self.output_path)

        if output_path.exists():
            output_path.unlink()

        laser_calibration.save(output_path)

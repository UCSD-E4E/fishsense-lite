from glob import glob
from pathlib import Path
from typing import List

import cv2
import fishsense_common.ray as ray
import matplotlib.pyplot as plt
import numpy as np
from fishsense_common.pluggable_cli import Command, argument
from pyaqua3ddev.image.image_processors import RawProcessor
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector
from pyfishsensedev.library.homography import viz2d
from pyfishsensedev.plane_detector.slate_detector import SlateDetector

from fishsense_lite.utils import uint16_2_uint8


@ray.remote(vram_mb=1536)
def execute(
    input_file: Path,
    lens_calibration: LensCalibration,
    estimated_laser_calibration: LaserCalibration,
    pdf: Pdf,
    debug_root: Path,
) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_path = debug_root / "field-calibration" / "laser"
    debug_path.mkdir(exist_ok=True, parents=True)

    png_name = input_file.name.replace("ORF", "PNG").replace("orf", "png")

    raw_processor = RawProcessor(enable_histogram_equalization=False)
    image_dark = uint16_2_uint8(raw_processor.load_and_process(input_file))

    image_rectifier = ImageRectifier(lens_calibration)
    image_dark = image_rectifier.rectify(image_dark)

    laser_detector = NNLaserDetector(
        lens_calibration, estimated_laser_calibration, device
    )
    laser_image_coord = laser_detector.find_laser(image_dark)

    if laser_image_coord is None:
        return None

    laser_detection_path = debug_path / f"detection_{png_name}"
    if laser_detection_path.exists():
        laser_detection_path.unlink()

    laser_detection = cv2.circle(
        image_dark.copy(),
        np.round(laser_image_coord).astype(int),
        radius=5,
        color=(0, 255, 0),
        thickness=-1,
    )
    cv2.imwrite(laser_detection_path.absolute().as_posix(), laser_detection)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(pdf.image)
    axarr[1].imshow(image_dark)
    f.savefig((debug_path / f"prematch_{png_name}"))
    f.show()

    cv2.imwrite((debug_path / f"dark_{png_name}").as_posix(), image_dark)

    slate_detector = SlateDetector(image_dark, pdf)
    if not slate_detector.is_valid():
        return None

    template_matches, image_matches = slate_detector._get_template_matches()

    plt.clf()
    viz2d.plot_images([pdf.image, image_dark])
    viz2d.plot_matches(template_matches, image_matches, color="lime", lw=0.2)
    viz2d.add_text(0, f"{len(template_matches)} matches", fs=20)
    plt.savefig((debug_path / f"matches_{png_name}"))

    laser_coord_3d = slate_detector.project_point_onto_plane_camera_space(
        laser_image_coord,
        lens_calibration.camera_matrix,
        lens_calibration.inverted_camera_matrix,
    )

    if np.any(np.isnan(laser_coord_3d)):
        return None

    return laser_coord_3d


class FieldCalibrateLaser(Command):
    @property
    def name(self) -> str:
        return "field-calibrate-laser"

    @property
    def description(self) -> str:
        return "Calibrates the laser for the FishSense Lite product line using the field calibration procedure."

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
        "--laser-position",
        short_name="-p",
        nargs=3,
        required=True,
        help="The laser position in centimeter inputed as x y z for the FishSense Lite product line.",
    )
    def laser_position(self) -> List[int]:
        return self.__laser_position

    @laser_position.setter
    def laser_position(self, value: List[int]):
        self.__laser_position = value

    @property
    @argument(
        "--laser-axis",
        short_name="-a",
        nargs=3,
        required=True,
        help="The laser axis unit vector inputed as x y z for the FishSense Lite product line.",
    )
    def laser_axis(self) -> List[float]:
        return self.__laser_axis

    @laser_axis.setter
    def laser_axis(self, value: List[float]):
        self.__laser_axis = value

    @property
    @argument(
        "--pdf",
        required=True,
        help="The PDF scan of a dive slate configured to be used for the FishSense Lite product line.",
    )
    def pdf(self) -> str:
        return self.__pdf

    @pdf.setter
    def pdf(self, value: str):
        self.__pdf = value

    @property
    @argument(
        "--output",
        short_name="-o",
        required=True,
        help="The path to store the resulting calibration.",
    )
    def output_path(self) -> str:
        return self.__output_path

    @output_path.setter
    def output_path(self, value: str):
        self.__output_path = value

    @property
    @argument("--overwrite", flag=True, help="Overwrite the calibration if it exists.")
    def overwrite(self) -> bool:
        return self.__overwrite

    @overwrite.setter
    def overwrite(self, value: bool):
        self.__overwrite = value

    @property
    @argument("--debug-path", help="Sets the debug path for storing debug images.")
    def debug_path(self) -> str:
        return self.__debug_path

    @debug_path.setter
    def debug_path(self, value: str):
        self.__debug_path = value

    def __init__(self):
        super().__init__()

        self.__data: List[str] = None
        self.__lens_calibration: str = None
        self.__laser_position: List[int] = None
        self.__laser_axis: List[float] = None
        self.__pdf: str = None
        self.__output_path: str = None
        self.__overwrite: bool = None
        self.__debug_path: str = None

    def __call__(self):
        self.init_ray()

        if self.debug_path is None:
            self.debug_path = ".debug"

        debug_path = Path(self.debug_path)

        files = [Path(f) for g in self.data for f in glob(g)]
        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        estimated_laser_calibration = LaserCalibration(
            np.array(self.laser_axis), np.array(self.laser_position)
        )

        pdf = Pdf(Path(self.pdf))

        futures = [
            execute.remote(
                f, lens_calibration, estimated_laser_calibration, pdf, debug_path
            )
            for f in files
        ]

        laser_points_3d = [
            p for p in self.tqdm(futures, total=len(files)) if p is not None
        ]
        laser_points_3d.sort(key=lambda x: x[2])
        laser_points_3d = np.array(laser_points_3d)

        laser_calibration = LaserCalibration()
        laser_calibration.plane_calibrate(
            laser_points_3d, estimated_laser_calibration, use_gauss_newton=False
        )

        output_path = Path(self.output_path)

        if output_path.exists() and self.overwrite:
            output_path.unlink()

        laser_calibration.save(output_path)

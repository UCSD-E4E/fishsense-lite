"""Module which represents the FishSense Lite Label Studio CLI."""

import importlib
import importlib.metadata
import json
from glob import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import fishsense_common.ray as ray
import numpy as np
import torch
from fishsense_common.pluggable_cli import Command, argument
from pyfishsensedev.calibration import LaserCalibration, LensCalibration
from pyfishsensedev.depth_map import DepthAnythingDepthMap, LaserDepthMap
from pyfishsensedev.image import ColorCorrection, ImageRectifier, RawProcessor
from pyfishsensedev.image.image_processors import RawProcessor
from pyfishsensedev.image.image_processors.raw_processor_old import RawProcessorOld
from pyfishsensedev.image.image_rectifier import ImageRectifier
from pyfishsensedev.laser.nn_laser_detector import NNLaserDetector
from pyfishsensedev.segmentation.fish.fish_segmentation_fishial_pytorch import (
    FishSegmentationFishialPyTorch,
)

from fishsense_lite.commands.label_studio_models.laser_label_studio_json import (
    LaserLabelStudioJSON,
)
from fishsense_lite.commands.label_studio_models.segmentation_label_studio_json import (
    SegmentationLabelStudioJSON,
)
from fishsense_lite.utils import get_output_file, get_root, uint16_2_uint8


@ray.remote(vram_mb=1200)
def execute_nn_laser(
    input_file: Path,
    lens_calibration: LensCalibration,
    estimated_laser_calibration: LaserCalibration,
    root: Path,
    output: Path,
    prefix: str,
    overwrite: bool,
) -> Tuple[np.ndarray, int, int]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_file = get_output_file(input_file, root, output, "jpg")
    json_file = output_file.with_suffix(".json")

    if output_file.exists() and json_file.exists() and not overwrite:
        return

    dark_raw_processor = RawProcessorOld(
        input_file, enable_histogram_equalization=False
    )
    try:
        image_dark = uint16_2_uint8(next(dark_raw_processor.__iter__()))
    except:
        return

    height, width, _ = image_dark.shape

    image_rectifier = ImageRectifier(lens_calibration)
    image_dark = image_rectifier.rectify(image_dark)

    laser_detector = NNLaserDetector(
        lens_calibration, estimated_laser_calibration, device
    )
    laser_image_coord = laser_detector.find_laser(image_dark)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_file.absolute().as_posix(), image_dark)

    json_objects = LaserLabelStudioJSON(
        prefix,
        output_file.relative_to(output.absolute()).as_posix(),
        laser_image_coord,
        width,
        height,
        laser_detector.name,
    )

    with open(json_file, "w") as f:
        f.write(json.dumps(json_objects, default=vars))


@ray.remote(vram_mb=768)
def execute_fishial(
    input_file: Path,
    lens_calibration: LensCalibration,
    estimated_laser_calibration: LaserCalibration,
    root: Path,
    output: Path,
    prefix: str,
    overwrite: bool,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_file = get_output_file(input_file, root, output, "jpg")
    mask_file = (
        output_file.parent
        / f"{output_file.name.removesuffix(output_file.suffix)}.mask.png"
    )
    json_file = output_file.with_suffix(".json")

    if output_file.exists() and json_file.exists() and not overwrite:
        return

    raw_processor = RawProcessor(input_file)
    dark_raw_processor = RawProcessorOld(
        input_file, enable_histogram_equalization=False
    )
    try:
        img = next(raw_processor.__iter__())
        img_dark = uint16_2_uint8(next(dark_raw_processor.__iter__()))
    except:
        return

    image_rectifier = ImageRectifier(lens_calibration)
    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = uint16_2_uint8(img)

    laser_detector = NNLaserDetector(
        lens_calibration, estimated_laser_calibration, device
    )
    laser_coords = laser_detector.find_laser(img_dark)

    ml_depth_map = DepthAnythingDepthMap(img8, device)

    if laser_coords is not None:
        laser_coords_int = np.round(laser_coords).astype(int)
        depth_map = LaserDepthMap(
            laser_coords, lens_calibration, estimated_laser_calibration
        )
        scale = (
            depth_map.depth_map[0, 0]
            / ml_depth_map.depth_map[laser_coords_int[1], laser_coords_int[0]]
        )
        ml_depth_map.rescale(scale)

    try:
        color_correction = ColorCorrection()
        img8 = uint16_2_uint8(color_correction.correct_color(img, ml_depth_map))
    except:
        pass

    fish_segmentation_inference = FishSegmentationFishialPyTorch(device)
    segmentations: np.ndarray = fish_segmentation_inference.inference(img8)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_file.absolute().as_posix(), img8)

    debug_output = (segmentations.astype(float) / segmentations.max() * 255).astype(
        np.uint8
    )
    cv2.imwrite(mask_file.absolute().as_posix(), debug_output)

    json_objects = SegmentationLabelStudioJSON(
        prefix,
        output_file.relative_to(output.absolute()).as_posix(),
        segmentations,
        fish_segmentation_inference.name,
    )

    with open(json_file, "w") as f:
        f.write(json.dumps(json_objects, default=vars))


class LabelStudioCommand(Command):
    """Command which represents the FishSense Lite Label Studio CLI."""

    @property
    def name(self) -> str:
        return "label-studio"

    @property
    def description(self) -> str:
        return "Outputs data in a format for comsuption with Label Studio."

    @property
    @argument("data", required=True, help="A glob that represents the data to process.")
    def data(self) -> List[str]:
        return self.__data

    @data.setter
    def data(self, value: List[str]):
        self.__data = value

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
        "--prefix",
        default="",
        help="The prefix to add to the output json file.",
    )
    def prefix(self) -> str:
        return self.__prefix

    @prefix.setter
    def prefix(self, value: str):
        self.__prefix = value

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
        self.__laser_position: List[int] = None
        self.__laser_axis: List[float] = None
        self.__output_path: str = None
        self.__prefix: str = None
        self.__overwrite: bool = None

    def __call__(self):
        self.init_ray()

        files = {Path(f).absolute() for g in self.data for f in glob(g, recursive=True)}
        root = get_root(files)

        lens_calibration = LensCalibration()
        lens_calibration.load(Path(self.lens_calibration))

        estimated_laser_calibration = LaserCalibration(
            np.array(self.laser_axis), np.array(self.laser_position)
        )

        output = Path(self.output_path)

        self.__build_nn_laser_json(
            files, lens_calibration, estimated_laser_calibration, root, output
        )

        self.__build_fishial_json(
            files, lens_calibration, estimated_laser_calibration, output, root
        )

    def __build_nn_laser_json(
        self,
        files: List[Path],
        lens_calibration: LensCalibration,
        estimated_laser_calibration: LaserCalibration,
        root: Path,
        output: Path,
    ):
        laser_detector = NNLaserDetector(
            lens_calibration, estimated_laser_calibration, "cpu"
        )

        output = (
            output
            / f"{laser_detector.name}.{importlib.metadata.version("pyfishsensedev")}"
        )
        output.mkdir(parents=True, exist_ok=True)

        futures = [
            execute_nn_laser.remote(
                f,
                lens_calibration,
                estimated_laser_calibration,
                root,
                output,
                self.prefix,
                self.overwrite,
            )
            for f in files
        ]

        list(self.tqdm(futures, total=len(files)))

    def __build_fishial_json(
        self,
        files: List[Path],
        lens_calibration: LensCalibration,
        estimated_laser_calibration: LaserCalibration,
        output: Path,
        root: Path,
    ):
        fish_segmentation = FishSegmentationFishialPyTorch("cpu")

        output = (
            output
            / f"{fish_segmentation.name}.{importlib.metadata.version("pyfishsensedev")}"
        )
        output.mkdir(parents=True, exist_ok=True)

        futures = [
            execute_fishial.remote(
                f,
                lens_calibration,
                estimated_laser_calibration,
                root,
                output,
                self.prefix,
                self.overwrite,
            )
            for f in files
        ]

        list(self.tqdm(futures, total=len(files)))

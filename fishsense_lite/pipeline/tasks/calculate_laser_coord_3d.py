import matplotlib.pyplot as plt
import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.calibration import LensCalibration
from pyfishsensedev.image.pdf import Pdf
from pyfishsensedev.plane_detector.slate_detector import SlateDetector
from skimage.util import img_as_ubyte


@task(output_name="laser_coord_3d")
def calculate_laser_coord_3d(
    img: np.ndarray[float],
    pdf: Pdf,
    laser_image_coords: np.ndarray[int],
    lens_calibration: LensCalibration,
) -> np.ndarray[float]:
    if pdf is None or laser_image_coords is None:
        return None

    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(pdf.image)
    # axarr[1].imshow(image_dark)
    # f.savefig((debug_path / f"prematch_{png_name}"))
    # f.show()

    # cv2.imwrite((debug_path / f"dark_{png_name}").as_posix(), image_dark)

    slate_detector = SlateDetector(img_as_ubyte(img), pdf)
    if not slate_detector.is_valid():
        return None

    template_matches, image_matches = slate_detector._get_template_matches()

    # plt.clf()
    # viz2d.plot_images([pdf.image, image_dark])
    # viz2d.plot_matches(template_matches, image_matches, color="lime", lw=0.2)
    # viz2d.add_text(0, f"{len(template_matches)} matches", fs=20)
    # plt.savefig((debug_path / f"matches_{png_name}"))

    laser_coord_3d = slate_detector.project_point_onto_plane_camera_space(
        laser_image_coords,
        lens_calibration.camera_matrix,
        lens_calibration.inverted_camera_matrix,
    )

    if np.any(np.isnan(laser_coord_3d)):
        return None

    return laser_coord_3d

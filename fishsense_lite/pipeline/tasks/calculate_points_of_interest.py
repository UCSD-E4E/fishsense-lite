from typing import Tuple

import numpy as np
from fishsense_common.pipeline.decorators import task
from pyfishsensedev.points_of_interest.points_of_interest_detector import (
    PointsOfInterestDetector,
)


@task(output_name="left_point, right_point")
def calculate_points_of_interest(
    points_of_interest_detector: PointsOfInterestDetector,
    segmentation_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    return points_of_interest_detector.find_points_of_interest(segmentation_mask)

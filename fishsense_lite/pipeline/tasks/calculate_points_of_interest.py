from typing import Tuple

import numpy as np
from fishsense_common.pipeline.decorators import task
from fishsense_common.pipeline.status import error, ok
from pyfishsensedev.points_of_interest.points_of_interest_detector import (
    PointsOfInterestDetector,
)


@task(output_name="left_point, right_point")
def calculate_points_of_interest(
    points_of_interest_detector: PointsOfInterestDetector,
    segmentation_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    left, right = points_of_interest_detector.find_points_of_interest(segmentation_mask)

    if left is None or right is None:
        error("INVALID_POINTS_OF_INTEREST")

    return ok((left, right))

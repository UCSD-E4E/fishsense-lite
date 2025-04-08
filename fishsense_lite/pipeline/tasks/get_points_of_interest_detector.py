from pathlib import Path

from fishsense_common.pipeline.decorators import task
from pyfishsensedev.points_of_interest.fish.label_studio_points_of_interest_detector import (
    LabelStudioPointsOfInterestDetector,
)
from pyfishsensedev.points_of_interest.points_of_interest_detector import (
    PointsOfInterestDetector,
)

from fishsense_lite.utils import PSqlConnectionString


# from pyfishsensedev.points_of_interest.fish
@task(output_name="points_of_interest_detector")
def get_points_of_interest_detector(
    input_file: Path,
    head_tail_labels_path: Path,
    use_sql_for_head_tail_labels: bool,
    connection_string: PSqlConnectionString,
) -> PointsOfInterestDetector:
    if head_tail_labels_path is not None:
        return LabelStudioPointsOfInterestDetector(input_file, head_tail_labels_path)
    # elif connection_string is not None and use_sql_for_head_tail_labels:
    #     return None

    raise NotImplementedError

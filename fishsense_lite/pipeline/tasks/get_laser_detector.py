from pathlib import Path

from fishsense_common.pipeline.decorators import task
from pyaqua3ddev.laser.single_laser.label_studio_laser_detector import (
    LabelStudioLaserDetector,
)
from pyaqua3ddev.laser.single_laser.laser_detector import LaserDetector
from pyaqua3ddev.laser.single_laser.psql_laser_detector import PSqlLabelDetector

from fishsense_lite.utils import PSqlConnectionString


@task(output_name="laser_detector")
def get_laser_detector(
    input_file: Path, laser_labels_path: Path, connection_string: PSqlConnectionString
) -> LaserDetector:
    if laser_labels_path is not None:
        return LabelStudioLaserDetector(input_file, laser_labels_path)
    elif connection_string is not None:
        return PSqlLabelDetector(
            input_file,
            connection_string.dbname,
            connection_string.username,
            connection_string.password,
            connection_string.host,
            port=connection_string.port,
        )

    raise NotImplementedError

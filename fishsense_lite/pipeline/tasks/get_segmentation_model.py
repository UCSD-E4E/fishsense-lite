from pathlib import Path

from fishsense_common.pipeline.decorators import task

from fishsense_lite.utils import PSqlConnectionString


@task(output_name="segmentation_model")
def get_segmentation_model(
    head_tail_labels_path: Path,
    use_sql_for_head_tail_labels: bool,
    connection_string: PSqlConnectionString,
) -> None:
    if head_tail_labels_path is not None:
        return None
    elif connection_string is not None and use_sql_for_head_tail_labels:
        return None

    raise NotImplementedError

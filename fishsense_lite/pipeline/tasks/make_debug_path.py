from pathlib import Path

from fishsense_common.pipeline.decorators import task


@task(output_name="debug_path")
def make_debug_path(debug_root: Path):
    debug_path = debug_root / "field-calibration" / "laser"
    debug_path.mkdir(exist_ok=True, parents=True)

    return debug_path

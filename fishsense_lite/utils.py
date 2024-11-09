from pathlib import Path
from typing import List

import numpy as np


def get_output_file(input_file: Path, root: Path, output: Path, extension: str) -> Path:
    return Path(
        input_file.absolute()
        .as_posix()
        .replace(root.as_posix(), output.absolute().as_posix())
        .replace(
            input_file.suffix,
            f".{extension}",
        )
    )


def get_root(files: List[Path]) -> Path | None:
    if not files:
        return None

    root = files
    while len(root) > 1:
        max_count = max(len(f.parts) for f in root)
        root = {f.parent if len(f.parts) == max_count else f for f in root}
    root = root.pop()

    return root


def uint16_2_double(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) / 65535


def uint16_2_uint8(img: np.ndarray) -> np.ndarray:
    return (uint16_2_double(img) * 255).astype(np.uint8)

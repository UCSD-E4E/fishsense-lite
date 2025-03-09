import hashlib
from pathlib import Path
from typing import List


def get_output_file(input_file: Path, root: Path, output: Path, extension: str) -> Path:
    hash = hashlib.md5(input_file.read_bytes()).hexdigest()
    return output / input_file.relative_to(root).parent / f"{hash}.{extension}"


def get_root(files: List[Path]) -> Path | None:
    if not files:
        return None

    root = files
    while len(root) > 1:
        max_count = max(len(f.parts) for f in root)
        root = {f.parent if len(f.parts) == max_count else f for f in root}
    root = root.pop()

    return root

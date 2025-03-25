import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlparse


@dataclass
class PSqlConnectionString:
    dbname: str
    username: str
    password: str
    host: str
    port: int


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


def parse_psql_connection_string(connection_string: str) -> PSqlConnectionString:
    if connection_string is not None:
        psql = urlparse(connection_string)
        dbname = psql.path[1:]
        username = (
            os.environ["PSQL_USERNAME"]
            if "PSQL_USERNAME" in os.environ
            else psql.username
        )
        password = (
            os.environ["PSQL_PASSWORD"]
            if "PSQL_PASSWORD" in os.environ
            else psql.password
        )
        host = psql.hostname
        port = 5432 if psql.port is None else psql.port

        return PSqlConnectionString(dbname, username, password, host, port)

    return None

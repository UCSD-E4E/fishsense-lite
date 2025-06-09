"""Define utility functions for the fishsense_lite package."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlparse


@dataclass
class PSqlConnectionString:
    """
    A dataclass to hold PostgreSQL connection string information.
    """

    dbname: str
    username: str
    password: str
    host: str
    port: int


def get_output_file(input_file: Path, root: Path, output: Path, extension: str) -> Path:
    """
    Get the output file path based on the input file, root directory, output directory,

    Args:
        input_file (Path): The input file.
        root (Path): The root directory.
        output (Path): The output directory.
        extension (str): The file extension for the output file.

    Returns:
        Path: The output file path.
    """

    parent_str = str(input_file.relative_to(root).parent)
    parent_str = parent_str[parent_str.index("://") + 3 :] if "://" in parent_str else parent_str

    hash_str = hashlib.md5(input_file.read_bytes()).hexdigest()
    return output / parent_str / f"{hash_str}.{extension}"


def get_root(files: List[Path]) -> Path | None:
    """
    Get the root directory of a list of files.

    Args:
        files (List[Path]): A list of Path objects representing the files.

    Returns:
        Path | None: The root directory of the files, or None if the list is empty.
    """
    if not files:
        return None

    root = files
    while len(root) > 1:
        max_count = max(len(f.parts) for f in root)
        root = {f.parent if len(f.parts) == max_count else f for f in root}
    root = root.pop()

    return root


def parse_psql_connection_string(connection_string: str) -> PSqlConnectionString:
    """
    Parse a PostgreSQL connection string and return a PSqlConnectionString object.
    The connection string should be in the format:
    postgresql://username:password@host:port/dbname
    If the connection string is None, return None.
    If the connection string is not in the correct format, raise a ValueError.
    The username and password can be overridden by the environment variables
    PSQL_USERNAME and PSQL_PASSWORD, respectively.

    Args:
        connection_string (str): The PostgreSQL connection string to parse.

    Returns:
        PSqlConnectionString: A PSqlConnectionString object containing the parsed
        connection string information.
    """
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

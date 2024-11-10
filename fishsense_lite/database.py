"""Module for accessing the sqlite database used to store the results."""

import datetime
import importlib
from os import makedirs
from pathlib import Path
from sqlite3 import Connection, Cursor, OperationalError, connect
from typing import Dict, Set

import backoff
import git

from fishsense_lite.result_status import ResultStatus


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._connection: Connection = None
        self._cursor: Cursor = None

    def __enter__(self):
        if not self._path.parent.exists():
            makedirs(self._path.parent.absolute().as_posix(), exist_ok=True)

        self._connection = connect(self._path)
        self._cursor = self._connection.cursor()

        self._create_metadata_table()
        self._insert_start_metadata()

        self._create_data_table()

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self._insert_end_metadata()

        self._cursor.close()
        self._connection.close()

        return True

    @backoff.on_exception(backoff.expo, OperationalError)
    def _create_metadata_table(self):
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS metadata
               (key text, value text)
            """
        )
        self._connection.commit()

    @backoff.on_exception(backoff.expo, OperationalError)
    def _create_data_table(self):
        self._cursor.execute(
            """CREATE TABLE IF NOT EXISTS data
               (time text, file text, result text, length float)
            """
        )
        self._connection.commit()

    @backoff.on_exception(backoff.expo, OperationalError)
    def _insert_start_metadata(self):
        repo = git.Repo(".")
        sha = repo.head.object.hexsha

        self.insert_metadata(
            {
                "start_time": datetime.datetime.now(datetime.UTC),
                "git_commit": sha,
                "version": importlib.metadata.version(),
            }
        )

        self._connection.commit()

    @backoff.on_exception(backoff.expo, OperationalError)
    def _insert_end_metadata(self):
        self.insert_metadata(
            {
                "end_time": datetime.datetime.now(datetime.UTC),
            }
        )

        self._connection.commit()

    @backoff.on_exception(backoff.expo, OperationalError)
    def insert_metadata(self, metadata: Dict[str, str]):
        self._cursor.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            list(metadata.items()),
        )
        self._connection.commit()

    @backoff.on_exception(backoff.expo, OperationalError)
    def insert_data(self, file: Path, result_status: ResultStatus, length: float):
        self._cursor.execute(
            "INSERT INTO data VALUES (?, ?, ?, ?)",
            (
                datetime.datetime.now(datetime.UTC),
                file.as_posix(),
                result_status.name,
                length,
            ),
        )
        self._connection.commit()

    @backoff.on_exception(backoff.expo, OperationalError)
    def get_files(self) -> Set[Path] | None:
        """Returns a set of Pathlib Paths which we have previously executed on.

        Returns:
            Set[Path] | None: A set of Pathlib paths we have previously executed on.
        """
        results = self._cursor.execute("SELECT file FROM data")

        if results:
            return {Path(row[0]) for row in results}

        return None

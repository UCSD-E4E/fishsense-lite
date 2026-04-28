"""End-to-end pg_dump test against the local devcontainer Postgres.

Verifies that `run_pg_dump` produces a real custom-format dump file
that `pg_restore -l` can parse. Avoids touching the NAS — that's the
activity-level test's job; this test only proves the pg_dump wrapper
itself works.
"""

import os
import subprocess
from pathlib import Path

import pytest

from fishsense_backup_worker.pg_dump import run_pg_dump


pytestmark = pytest.mark.integration


def _pg_host() -> str:
    return os.environ.get("FISHSENSE_POSTGRES_HOST", "postgres")


def _pg_port() -> int:
    return int(os.environ.get("FISHSENSE_POSTGRES_PORT", "5432"))


def _pg_user() -> str:
    return os.environ.get("FISHSENSE_POSTGRES_USER", "postgres")


def _pg_password() -> str:
    return os.environ.get("FISHSENSE_POSTGRES_PASSWORD", "fishsense_local")


def _pg_db() -> str:
    return os.environ.get("FISHSENSE_POSTGRES_DB", "fishsense")


def test_run_pg_dump_produces_a_pg_restore_listable_dump(tmp_path: Path):
    out = tmp_path / "fishsense.dump"
    run_pg_dump(
        db_name=_pg_db(),
        host=_pg_host(),
        port=_pg_port(),
        username=_pg_user(),
        password=_pg_password(),
        output_path=str(out),
        timeout_s=300.0,
    )

    assert out.exists() and out.stat().st_size > 1024, (
        f"dump file is missing or suspiciously small: {out}"
    )

    # `pg_restore -l <file>` parses the table-of-contents of a custom-
    # format archive. If the file isn't a valid PG custom dump, this
    # exits non-zero. This is a stronger check than just "non-empty".
    result = subprocess.run(
        ["pg_restore", "-l", str(out)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The TOC always lists at least one entry for a non-empty schema.
    assert result.stdout.strip(), "pg_restore -l returned empty TOC"

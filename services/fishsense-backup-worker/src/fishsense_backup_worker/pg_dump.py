"""Subprocess wrapper around `pg_dump`.

The command builder is broken out from the runner so it can be unit-
tested without invoking the real binary, and so the password-via-env
contract is locked down by tests (passwords in argv would leak into
`ps` output)."""

import logging
import subprocess
from typing import Dict, List, Tuple

_log = logging.getLogger(__name__)


def build_pg_dump_command(
    *,
    db_name: str,
    host: str,
    port: int,
    username: str,
    password: str,
    output_path: str,
) -> Tuple[List[str], Dict[str, str]]:
    """Return (argv, env) for the pg_dump invocation.

    Format is `-Fc` (custom) — what `pg_restore` expects and what the
    existing prod backup convention uses. Password goes in PGPASSWORD,
    NOT argv.
    """
    cmd = [
        "pg_dump",
        "-Fc",
        "-h", host,
        "-p", str(port),
        "-U", username,
        "-d", db_name,
        "-f", output_path,
    ]
    env = {"PGPASSWORD": password}
    return cmd, env


def run_pg_dump(
    *,
    db_name: str,
    host: str,
    port: int,
    username: str,
    password: str,
    output_path: str,
    timeout_s: float = 3600.0,
) -> None:
    """Invoke pg_dump synchronously. Raises CalledProcessError on
    non-zero exit. Caller is responsible for cleaning up the output
    file on failure (we leave the partial dump in place so it's
    available for postmortem)."""
    cmd, env = build_pg_dump_command(
        db_name=db_name,
        host=host,
        port=port,
        username=username,
        password=password,
        output_path=output_path,
    )
    _log.info(
        "pg_dump start db=%s host=%s:%d user=%s out=%s",
        db_name, host, port, username, output_path,
    )
    subprocess.run(
        cmd,
        env=env,
        check=True,
        timeout=timeout_s,
    )
    _log.info("pg_dump done db=%s out=%s", db_name, output_path)

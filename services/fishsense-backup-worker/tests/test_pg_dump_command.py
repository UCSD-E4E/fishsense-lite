"""Pure-logic tests for the pg_dump command builder.

Constructs the exact argv + env passed to subprocess. The password
must NOT appear in argv (it goes via PGPASSWORD env), and the format
flag must be `-Fc` (custom format — what the prod backup convention
uses, what `pg_restore` expects).
"""

import subprocess
from subprocess import CompletedProcess

import pytest

from fishsense_backup_worker.pg_dump import build_pg_dump_command, run_pg_dump


def test_uses_custom_format():
    cmd, _env = build_pg_dump_command(
        db_name="fishsense",
        host="postgres",
        port=5432,
        username="backup_user",
        password="hunter2",
        output_path="/tmp/out.dump",
    )
    assert "-Fc" in cmd or "--format=custom" in cmd


def test_passes_db_name_via_dash_d():
    cmd, _ = build_pg_dump_command(
        db_name="superset",
        host="postgres",
        port=5432,
        username="backup_user",
        password="hunter2",
        output_path="/tmp/out.dump",
    )
    # Either `-d superset` or `--dbname=superset` is acceptable; pin to
    # one shape so future readers don't have to re-derive it.
    assert "-d" in cmd
    assert cmd[cmd.index("-d") + 1] == "superset"


def test_passes_connection_args():
    cmd, _ = build_pg_dump_command(
        db_name="fishsense",
        host="postgres.internal",
        port=5433,
        username="backup_user",
        password="hunter2",
        output_path="/tmp/out.dump",
    )
    assert "-h" in cmd and cmd[cmd.index("-h") + 1] == "postgres.internal"
    assert "-p" in cmd and cmd[cmd.index("-p") + 1] == "5433"
    assert "-U" in cmd and cmd[cmd.index("-U") + 1] == "backup_user"


def test_password_goes_in_env_not_argv():
    """argv is visible in `ps`; the password must not appear there."""
    cmd, env = build_pg_dump_command(
        db_name="fishsense",
        host="postgres",
        port=5432,
        username="backup_user",
        password="extremely-secret",
        output_path="/tmp/out.dump",
    )
    assert "extremely-secret" not in cmd
    assert env.get("PGPASSWORD") == "extremely-secret"


def test_writes_to_specified_output_path():
    cmd, _ = build_pg_dump_command(
        db_name="fishsense",
        host="postgres",
        port=5432,
        username="backup_user",
        password="hunter2",
        output_path="/var/backups/fishsense.dump",
    )
    assert "-f" in cmd
    assert cmd[cmd.index("-f") + 1] == "/var/backups/fishsense.dump"


def test_first_arg_is_pg_dump_executable():
    cmd, _ = build_pg_dump_command(
        db_name="fishsense",
        host="postgres",
        port=5432,
        username="backup_user",
        password="hunter2",
        output_path="/tmp/out.dump",
    )
    assert cmd[0] == "pg_dump"


def test_failure_surfaces_stderr_in_exception(monkeypatch, tmp_path):
    """A non-zero pg_dump exit must raise CalledProcessError with the
    captured stderr attached, so Temporal failure events carry the
    real reason (auth, role missing, permission denied) instead of
    just the exit code."""

    def fake_run(cmd, **kwargs):
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        return CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr='pg_dump: error: connection to server at "postgres" '
            'failed: FATAL:  password authentication failed for user "backup"\n',
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_pg_dump(
            db_name="fishsense",
            host="postgres",
            port=5432,
            username="backup",
            password="wrong",
            output_path=str(tmp_path / "out.dump"),
        )

    assert excinfo.value.returncode == 1
    assert "password authentication failed" in (excinfo.value.stderr or "")


def test_env_does_not_set_other_pg_vars():
    """We pass connection args explicitly via -h/-p/-U/-d, so leaking
    a stray PGHOST or PGDATABASE from the parent env into the command
    env could change behavior surprisingly. The builder should return
    only PGPASSWORD."""
    _, env = build_pg_dump_command(
        db_name="fishsense",
        host="postgres",
        port=5432,
        username="backup_user",
        password="hunter2",
        output_path="/tmp/out.dump",
    )
    assert set(env) == {"PGPASSWORD"}

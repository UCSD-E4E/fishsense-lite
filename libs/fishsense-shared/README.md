# fishsense-shared

Shared helpers used by every FishSense Lite service. Workspace-only — not
published.

## What's in here

| Module | Surface | Notes |
|---|---|---|
| `config` | `IS_DOCKER`, `get_config_path()`, `get_log_path(app_name)`, `path_validator`, `url_condition` | Picks `/e4efs/{config,logs}` when `E4EFS_DOCKER=true`, else cwd / platformdirs. `get_config_path()` falling back to cwd is what causes the repo-root `settings.toml` side-effect — see `CLAUDE.md`. `url_condition` is the lenient http(s)-with-non-empty-hostname validator used by every service's `*.url` setting (see CLAUDE.md for why we don't use `validators.url`). |
| `logging` | `configure_logging(app_name, log_filename=None)`, `configure_log_handler(handler)` | Root logger at DEBUG, UTC timestamps, midnight-rotated file handler + console handler. |
| `temporal` | `build_tls_config(temporal_settings)`, `ensure_schedule(client, schedule_id, schedule)` | `build_tls_config` reads cert files referenced by `settings.temporal` and returns a `TLSConfig` (or `None` when `temporal.tls=False`). `ensure_schedule` is the idempotent schedule-creation helper used by both the api-worker and the backup-worker — refuses to update in place; an operator must `temporal schedule delete <id>` to change config. |
| `exception_group` | `ExceptionGroupErrorLogging(logger)` | Async context manager that flattens `ExceptionGroup`s from `asyncio.TaskGroup` into per-exception log lines. |

## Conventions enforced

- **`E4EFS_` envvar prefix.** Every service uses `Dynaconf(envvar_prefix="E4EFS", ...)`. The config helpers here assume that.
- **Settings files.** Services load `(get_config_path() / "settings.toml")` and `(get_config_path() / ".secrets.toml")`. Outside Docker that resolves to cwd, so don't run a worker from the repo root unless you want a `settings.toml` to materialize there.
- **Temporal mTLS shape.** `build_tls_config` expects `settings.temporal.{tls, client_cert, client_private_key}`, plus optional `server_root_ca_cert` and `domain`. New workers should reuse the same key names so this helper keeps applying.

## Adding to the public surface

Re-export from `fishsense_shared/__init__.py` and append to `__all__`. Keep
this lib import-cheap — it's loaded eagerly by every service's
`config.py` at module level (Dynaconf validates on first attribute
access; see the CLAUDE.md gotcha).

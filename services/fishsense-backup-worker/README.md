# fishsense-backup-worker

Scheduled Postgres → NAS backup worker.

Runs `pg_dump -Fc` nightly at 03:00 UTC for the configured databases
(default: `fishsense`, `superset`, `temporal_db`), uploads each dump to
the NAS under `{nas_root}/{db_name}/{ISO8601}.dump`, then prunes anything
beyond the last 14.

Deliberately separate from the data-processing-workflow-worker —
narrower blast radius (only `pg_dump`-equivalent DB credentials, only
NAS write access), separate task queue, separate image.

The schedule is auto-created by the worker on startup if missing
(idempotent), so the first deploy of this service kicks off the cadence
without manual ops.

## Required env (E4EFS_ prefix)

- `E4EFS_TEMPORAL__HOST`, `E4EFS_TEMPORAL__PORT` — temporal cluster
- `E4EFS_E4E_NAS__URL`, `E4EFS_E4E_NAS__USERNAME`, `E4EFS_E4E_NAS__PASSWORD` — NAS access
- `E4EFS_POSTGRES__HOST`, `E4EFS_POSTGRES__PORT`, `E4EFS_POSTGRES__USERNAME`, `E4EFS_POSTGRES__PASSWORD` — DB access (recommend a dedicated `backup` role with the minimum perms `pg_dump` needs)

## Optional config (settings.toml)

```toml
[backup]
databases = ["fishsense", "superset", "temporal_db"]
retention_count = 14
nas_root_path = "/fishsense_backups"
schedule_id = "fishsense-daily-db-backup"
schedule_cron = "0 3 * * *"  # daily 03:00 UTC
```

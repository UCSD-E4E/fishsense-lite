# Postgres init scripts — intentionally empty (restore-based bootstrap)

The Incus deploy bootstraps Postgres by **restoring a prod `pg_dump`** into the
`pgdata` named volume, not by running init SQL. So this directory ships **no
`*.sql`/`*.sh`** — Postgres' official image only runs `/docker-entrypoint-initdb.d/`
on a *fresh* data dir anyway, and after a restore the dir is non-empty.

## Cutover procedure (operator)

1. Bring up **only** postgres against a fresh `pgdata` volume, or stop the stack.
2. Restore the latest prod dump (the nightly `fishsense-backup-worker` output) —
   this brings the `fishsense` + `superset` databases **and** the DB roles
   (`postgres` admin, `superset`, `backup`) **with their existing passwords**:
   ```
   pg_restore -U postgres -d postgres --clean --create <prod_dump>
   ```
   (Skip `temporal_db` / `temporal_visibility` — Temporal is external now, krg-prod.)
   **This is also the Postgres 16 → 17 major upgrade** — restoring an older dump into
   the newer server is the supported dump/restore upgrade path. The backup worker's
   `pg_dump` client is pg17 (Debian trixie), so ongoing dumps of the 17 server are fine.
3. Seed OpenBao `secret/tenants/fishsense/*` with **those same** role passwords
   (`postgres.password`, `superset.db_password`, `postgres.backup_password`) so the
   vault-agent render (`app.env`) matches what the restore created. See
   [`../../secrets.nix`](../../secrets.nix).
4. Bring the full stack up.

## Fresh (no-data) env

If you ever stand up an empty instance (dev / DR with no dump), you must create the
`superset` DB + role (read-only SELECT on `fishsense`) and the `backup` role
(`pg_read_all_data`) by hand — see the old stack's
`deploy/pg_volumes/scripts/2025-09-02_create_database.sql` +
`2026-05-01_create_backup_role.sql` for the exact grants. Passwords must match what
you seed in OpenBao. This is deliberately not automated here to avoid an empty-DB
footgun on the restore path.

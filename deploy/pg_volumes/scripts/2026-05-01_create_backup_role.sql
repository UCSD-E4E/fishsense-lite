-- Backup-worker postgres role.
--
-- pg_dump needs SELECT on every table in every database it dumps, plus
-- USAGE on each schema. pg_read_all_data (PG14+) covers exactly that
-- without us having to hand-grant per-table.
--
-- This script only runs on fresh data volumes (Postgres official image
-- ignores /docker-entrypoint-initdb.d/ when PGDATA already exists). For
-- existing clusters (i.e. prod), an operator must run the equivalent
-- statements by hand using the credentials in
-- backup_worker_volumes/config/.secrets.toml on the host.
--
-- Replace the SCRAM-SHA-256 placeholder below with a hash generated
-- from the password in .secrets.toml. To generate the hash:
--   psql -U postgres -c "SELECT 'SCRAM-SHA-256$' || ..."
-- or simply use plaintext at create time and let Postgres encode it,
-- then `\password backup` to rotate (matching how the temporal/superset
-- roles in the sibling files were seeded).

CREATE ROLE backup WITH
    LOGIN
    ENCRYPTED PASSWORD 'REPLACE_WITH_SCRAM_HASH_FROM_SECRETS_TOML';

GRANT pg_read_all_data TO backup;

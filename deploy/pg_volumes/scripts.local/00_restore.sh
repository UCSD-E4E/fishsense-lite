#!/usr/bin/env bash
# Runs once on first init of the postgres container (i.e. when the named
# volume `pg_data` is empty). On subsequent boots Postgres skips
# /docker-entrypoint-initdb.d/ entirely.
#
# Responsibilities:
#   1. Create the `temporal` role used by the temporal service.
#   2. Create the empty fishsense + temporal databases.
#   3. Restore the prod dump into the fishsense database (if mounted).
#
# The dump is bind-mounted at /var/backups/fishsense.dump from the host path
# in $FISHSENSE_DUMP_PATH. If the user hasn't pointed that at a real file the
# compose.local.yml falls back to /dev/null and we skip the restore — the DB
# is still usable, just empty.

set -euo pipefail

DB_NAME="${FISHSENSE_DB_NAME:-fishsense}"
TEMPORAL_USER="${TEMPORAL_DB_USER:-temporal}"
TEMPORAL_PWD="${TEMPORAL_DB_PWD:-temporal_local}"
DUMP_PATH="/var/backups/fishsense.dump"

echo "[init] creating temporal role + databases"
psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname postgres <<SQL
CREATE ROLE ${TEMPORAL_USER} WITH LOGIN PASSWORD '${TEMPORAL_PWD}' CREATEDB;
CREATE DATABASE ${DB_NAME} OWNER ${POSTGRES_USER};
SQL

# temporalio/auto-setup creates temporal_db + temporal_visibility_db itself
# (SKIP_DB_CREATE is not set in compose.local.yml), so we don't pre-create
# those here.

if [ ! -s "${DUMP_PATH}" ]; then
    echo "[init] no dump at ${DUMP_PATH} (or empty) — leaving ${DB_NAME} empty."
    echo "[init] set FISHSENSE_DUMP_PATH in deploy/.env to restore prod data."
    exit 0
fi

# pg_dump custom format → pg_restore. --no-owner / --no-privileges drops the
# prod role assignments so the restore works against a fresh local DB owned
# by `postgres`.
echo "[init] restoring ${DUMP_PATH} → ${DB_NAME} (this can take a few minutes)"
pg_restore \
    --username "${POSTGRES_USER}" \
    --dbname "${DB_NAME}" \
    --no-owner \
    --no-privileges \
    --exit-on-error \
    "${DUMP_PATH}"
echo "[init] restore complete"

# fishsense-backup-worker
#
# IMPORTANT: build with the **repo root** as context (see compose +
# CI), not this service directory. uv.lock lives at the repo root and
# the workspace resolution needs every member's pyproject.toml.
#
# Example:
#   docker build -t fishsense-backup-worker \
#     -f services/fishsense-backup-worker/Dockerfile .
FROM python:3.13-slim-trixie

# `pg_dump` (postgresql-client) is the load-bearing dep. ca-certificates
# for HTTPS to NAS + Temporal. Pinned-version-free apt to keep the image
# slim; the postgres major must match (or exceed) the prod server major
# — bump if/when the prod cluster moves past PG17.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade uv

WORKDIR /app

# Copy the workspace root manifest + lockfile first so dependency
# resolution caches across source-only edits.
COPY pyproject.toml uv.lock /app/

# Workspace member pyproject.toml files for resolution.
COPY libs/fishsense-shared/pyproject.toml /app/libs/fishsense-shared/pyproject.toml
COPY services/fishsense-backup-worker/pyproject.toml /app/services/fishsense-backup-worker/pyproject.toml

# Source trees (only what fishsense-backup-worker actually depends on).
COPY libs/fishsense-shared /app/libs/fishsense-shared
COPY services/fishsense-backup-worker /app/services/fishsense-backup-worker

# Sync only the backup-worker package + its workspace deps; --frozen
# enforces the committed lockfile.
RUN uv sync --frozen --package fishsense-backup-worker --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

CMD ["fishsense_backup_worker"]

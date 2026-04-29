# fishsense-backup-worker
#
# IMPORTANT: build with the **repo root** as context. uv.lock lives at
# repo root and the workspace resolution needs every member's
# pyproject.toml.
#
#   docker build -t fishsense-backup-worker \
#     -f services/fishsense-backup-worker/Dockerfile .

FROM ghcr.io/astral-sh/uv:python3.13-trixie AS builder

WORKDIR /app

# Workspace root manifest + lockfile.
COPY pyproject.toml uv.lock /app/

# Every workspace member's pyproject.toml — uv needs all of them present
# to satisfy `[tool.uv.workspace] members`.
COPY libs/fishsense-api-sdk/pyproject.toml /app/libs/fishsense-api-sdk/pyproject.toml
COPY libs/fishsense-shared/pyproject.toml /app/libs/fishsense-shared/pyproject.toml
COPY services/fishsense-api/pyproject.toml /app/services/fishsense-api/pyproject.toml
COPY services/fishsense-api-workflow-worker/pyproject.toml /app/services/fishsense-api-workflow-worker/pyproject.toml
COPY services/fishsense-backup-worker/pyproject.toml /app/services/fishsense-backup-worker/pyproject.toml
COPY services/fishsense-data-processing-workflow-worker/pyproject.toml /app/services/fishsense-data-processing-workflow-worker/pyproject.toml

# Workspace dep sources (uv needs these to build their wheels under
# --no-editable) + this package's source.
COPY libs/fishsense-shared /app/libs/fishsense-shared
COPY services/fishsense-backup-worker /app/services/fishsense-backup-worker

# Single sync — install runtime deps + the package.
RUN uv sync --frozen --no-dev --no-editable \
        --package fishsense-backup-worker \
    && uv cache clean

# --- Runtime ---
FROM python:3.13-slim-trixie AS runtime

# pg_dump is the load-bearing dep; ca-certificates is for HTTPS to the
# NAS / Temporal cluster. Postgres major must >= the prod server major;
# bump if the prod cluster ever moves past PG17.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        postgresql-client \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV E4EFS_DOCKER=true

RUN mkdir -p /e4efs/config /e4efs/logs /e4efs/data /e4efs/cache

COPY --from=builder /app/.venv /app/.venv

ENTRYPOINT ["fishsense_backup_worker"]

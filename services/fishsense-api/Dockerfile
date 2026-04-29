# fishsense-api (FastAPI app)
#
# IMPORTANT: build with the **repo root** as context. uv.lock lives at
# repo root and the workspace resolution needs every member's
# pyproject.toml.
#
#   docker build -t fishsense-api \
#     -f services/fishsense-api/Dockerfile .

FROM ghcr.io/astral-sh/uv:python3.13-trixie AS builder

WORKDIR /app

COPY pyproject.toml uv.lock /app/

# Every workspace member's pyproject.toml — uv needs all of them present
# to satisfy `[tool.uv.workspace] members`.
COPY libs/fishsense-api-sdk/pyproject.toml /app/libs/fishsense-api-sdk/pyproject.toml
COPY libs/fishsense-shared/pyproject.toml /app/libs/fishsense-shared/pyproject.toml
COPY services/fishsense-api/pyproject.toml /app/services/fishsense-api/pyproject.toml
COPY services/fishsense-api-workflow-worker/pyproject.toml /app/services/fishsense-api-workflow-worker/pyproject.toml
COPY services/fishsense-backup-worker/pyproject.toml /app/services/fishsense-backup-worker/pyproject.toml
COPY services/fishsense-data-processing-workflow-worker/pyproject.toml /app/services/fishsense-data-processing-workflow-worker/pyproject.toml

# Workspace dep source (fishsense-shared) + this package's source.
# fishsense-api-sdk is a dev-only dep here, so omitted from runtime.
COPY libs/fishsense-shared /app/libs/fishsense-shared
COPY services/fishsense-api /app/services/fishsense-api

# Install runtime deps + the package.
RUN uv sync --frozen --no-dev --no-editable \
        --package fishsense-api \
    && uv cache clean

# --- Runtime ---
FROM python:3.13-slim-trixie AS runtime

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV E4EFS_DOCKER=true

RUN mkdir -p /e4efs/config /e4efs/logs /e4efs/data /e4efs/cache

COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000
ENTRYPOINT ["uvicorn", "fishsense_api:app", "--host", "0.0.0.0", "--port", "8000"]

# fishsense-data-processing-workflow-worker (Temporal worker — data-side)
#
# IMPORTANT: build with the **repo root** as context.
#
#   docker build -t fishsense-data-processing-workflow-worker \
#     -f services/fishsense-data-processing-workflow-worker/Dockerfile .

FROM ghcr.io/astral-sh/uv:python3.13-trixie AS builder

WORKDIR /app

COPY pyproject.toml uv.lock /app/

COPY libs/fishsense-api-sdk/pyproject.toml /app/libs/fishsense-api-sdk/pyproject.toml
COPY libs/fishsense-shared/pyproject.toml /app/libs/fishsense-shared/pyproject.toml
COPY services/fishsense-api/pyproject.toml /app/services/fishsense-api/pyproject.toml
COPY services/fishsense-api-workflow-worker/pyproject.toml /app/services/fishsense-api-workflow-worker/pyproject.toml
COPY services/fishsense-backup-worker/pyproject.toml /app/services/fishsense-backup-worker/pyproject.toml
COPY services/fishsense-data-processing-workflow-worker/pyproject.toml /app/services/fishsense-data-processing-workflow-worker/pyproject.toml

# data-worker imports from fishsense-api-sdk + fishsense-shared.
# fishsense-core is a prebuilt cp313 manylinux wheel pinned in this
# package's pyproject; no build-time C deps needed in the builder stage.
COPY libs/fishsense-api-sdk /app/libs/fishsense-api-sdk
COPY libs/fishsense-shared /app/libs/fishsense-shared
COPY services/fishsense-data-processing-workflow-worker /app/services/fishsense-data-processing-workflow-worker

RUN uv sync --frozen --no-dev --no-editable \
        --package fishsense-data-processing-workflow-worker \
    && uv cache clean

# --- Runtime ---
FROM python:3.13-slim-trixie AS runtime

# OpenCV runtime libs — same set the devcontainer pulls in (libgl1 +
# libglib2.0-0 are what opencv-python links against). ca-certificates
# for HTTPS to fishsense-api / static_file_server / NAS.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV E4EFS_DOCKER=true

RUN mkdir -p /e4efs/config /e4efs/logs /e4efs/data /e4efs/cache

COPY --from=builder /app/.venv /app/.venv

ENTRYPOINT ["fishsense_data_processing_workflow_worker"]

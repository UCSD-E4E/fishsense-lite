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
# libglib2.0-0 are what opencv-python links against). libgomp1 is torch's
# OpenMP runtime (pulled in by the [laser-detector] extra). ca-certificates
# for HTTPS to fishsense-api / static_file_server / NAS / HuggingFace, and
# curl to fetch the baked laser-detector checkpoint. torch's CUDA runtime is
# vendored inside the wheel, so no CUDA base image is needed — the NRP GPU
# node supplies only the driver.
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ca-certificates \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV E4EFS_DOCKER=true

RUN mkdir -p /e4efs/config /e4efs/logs /e4efs/data /e4efs/cache /e4efs/models

# Bake the laser-detector checkpoint into the image so the GPU predict
# activity never has to reach HuggingFace at run time (preemptible NRP pods,
# no HF token). Public repo. `E4EFS_LASER_DETECTOR__CHECKPOINT` in the activity
# defaults to this path. run3 = the production ResNet-34 checkpoint whose
# fp32 bias-offset is calibrated in fishsense-core.
RUN curl -fsSL \
        "https://huggingface.co/ucsde4e/fishsense-laser-detector/resolve/main/run3_epoch_021.pt" \
        -o /e4efs/models/run3_epoch_021.pt

COPY --from=builder /app/.venv /app/.venv

ENTRYPOINT ["fishsense_data_processing_workflow_worker"]

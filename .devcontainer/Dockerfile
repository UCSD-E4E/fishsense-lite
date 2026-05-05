FROM mcr.microsoft.com/devcontainers/python:3.13-trixie

# fishsense-core ships as a prebuilt cp313 manylinux wheel, so we don't need
# its build-time deps (libopencv-dev, libopenblas-dev, libclang). We only need
# runtime libs that opencv-python links against.
#
# docker-cli + docker-compose-v2 + git let the dev container drive the local
# stack itself (e.g. `docker compose -f deploy/compose.local.yml up -d
# --force-recreate <svc>`) by bind-mounting the host's docker socket. The
# `vscode` user gets access to the socket via the `group_add` entry on the
# `dev` service in compose.local.yml — no in-container daemon is started.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        postgresql-client \
        docker-cli \
        docker-compose \
        ca-certificates \
        curl \
        gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Node 24 (current LTS) for apps/web. Trixie's default `nodejs` apt is v18 —
# too old for Next.js 15 + React 19. Pull from NodeSource to track Node 24.x.
# npm ships in the same package.
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv is the project's package manager. Install for the vscode user so that
# postCreateCommand `uv sync` lands the venv where Pylance expects it.
RUN pip install --no-cache-dir --upgrade uv

# Pre-create the uv cache dir owned by vscode. Docker initializes a named
# volume mounted onto an existing directory by copying that directory's
# ownership/perms — so this is what gives the `dev_uv_cache` volume the right
# perms on first creation.
RUN mkdir -p /home/vscode/.cache/uv \
    && chown -R vscode:vscode /home/vscode/.cache

# Drop into the workspace as the non-root vscode user (provided by the base
# image). The repo is bind-mounted at the host's absolute path
# (HOST_REPO_PATH from deploy/.env), so we don't pin a WORKDIR here — the
# `dev` service in compose.local.yml sets `working_dir: ${HOST_REPO_PATH}`,
# and VSCode overrides the cwd to `workspaceFolder` on attach.
USER vscode
ENV PATH="/home/vscode/.local/bin:${PATH}"

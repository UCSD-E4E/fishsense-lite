# fishsense-lite

UCSD E4E FishSense Lite system monorepo. uv workspace consolidated from the
prior polyrepo split.

## Repository layout

Services (each is a workspace member with its own Dockerfile under
`services/<name>/Dockerfile`, built with the **repo root** as context):

- `services/fishsense-api/` — FastAPI server in front of PostgreSQL
- `services/fishsense-api-workflow-worker/` — Temporal worker, NAS + Label
  Studio orchestration (api side)
- `services/fishsense-data-processing-workflow-worker/` — Temporal worker
  for per-image preprocessing (rectify, overlay, JPEG, file-exchange PUT)
- `services/fishsense-backup-worker/` — Temporal worker for nightly
  Postgres → NAS backups + retention pruning

Libraries (workspace members, not published separately):

- `libs/fishsense-shared/` — shared Dynaconf / TLS / logging helpers
- `libs/fishsense-api-sdk/` — Python HTTP client SDK (was an external repo
  before the monorepo cutover)

Deploy:

- `deploy/compose.yml` + `compose.{orchestrator,superset,temporal,workers}.yml`
  — prod docker-compose stack (formerly the `fishsense-web-services` repo)
- `deploy/compose.local.yml` — self-contained local devcontainer stack
  (postgres + temporal + fishsense-api + nginx static_file_server)

## Local development

Open in VSCode and "Reopen in Container". The devcontainer brings up the
local stack via `deploy/compose.local.yml`. One-time setup:

1. `cp deploy/.env.local.example deploy/.env`
2. Edit `HOST_REPO_PATH` (output of `realpath .` from the repo root),
   `DOCKER_GID` (`getent group docker | cut -d: -f3`), and optionally
   `FISHSENSE_DUMP_PATH` to point at a prod backup so the local postgres
   restores from it.

`uv sync --all-packages --all-groups` happens automatically on container
create. The fastest way to verify a change before pushing is the top-level
`check.sh` script, which mirrors what CI runs:

```
./check.sh lint         # pylint on Python files changed since origin/main
./check.sh unit         # pytest with default markers across all packages
./check.sh integration  # pytest -m integration (needs the local stack up)
./check.sh all          # lint, then unit, then integration
```

To invoke pytest directly against a specific package:

```
uv run --package <pkg> python -m pytest <path>            # unit tests (default)
uv run --package <pkg> python -m pytest <path> -m integration  # opts in
```

Use `python -m pytest`, not bare `pytest` — the bare entry-point picks up a
different `sys.path` and fails to import workspace packages.

## CI / deploy pipeline

Four GitHub Actions workflows under `.github/workflows/`:

```
push to main         → build.yml    (image -> :sha-<short> + :main)
release-please merge → release.yml  (cuts GitHub release + tag)
release: published   → promote.yml  (:sha-<short> -> :v<version>; opens
                                      auto-deploy/* PR bumping deploy/
                                      compose*.yml pin)
auto-deploy PR merge → deploy.yml   (docker compose pull && up -d on
                                      self-hosted [fishsense-prod] runner)
```

Build-once / promote-tag means every release ships the *exact* image that
was built from the release commit, regardless of any newer non-release
commits that landed on main. Deploy is intentional — a human reviews the
version-pin diff in the auto-deploy PR before prod restarts.

`deploy.yml` operates on a persistent ops-managed deploy directory on the
host (path supplied via repo variable `DEPLOY_DIR`); volumes and secrets
sit there as untracked siblings of the compose files. See
`.github/workflows/deploy.yml` header for the host-bootstrap steps.

## External dependencies

Consumed via released wheels / git refs, not part of this monorepo:

- [fishsense-core](https://github.com/UCSD-E4E/fishsense-core) — Rust + PyO3
  compute library (rectification, world-point projection, laser calibration)

## Where to read next

- **[CLAUDE.md](CLAUDE.md)** — the operating manual. Service map,
  notebook-port status, data-worker activity pattern, file-exchange
  URL contract, Dynaconf eager-validation gotcha, build → release →
  promote → deploy pipeline, service Dockerfile pattern, **and the
  operational ground truth** (no staging tier, authentik in front of
  the API, `E4EFS_` env prefix, SDK ↔ API drift policing) plus the
  open follow-ups (migration finding #1, phase-6 polyrepo cutover,
  stage14 real-frame regression, Label Studio bootstrap chicken-and-egg).
  **Read this before editing.**
- **[docs/diagrams.md](docs/diagrams.md)** — Mermaid diagrams: system
  context, deploy topology, domain model, SDK class diagram, per-stage
  sequence flows, CI/CD state machine. Useful as a high-level
  orientation.
- **Per-package READMEs** in [services/*/README.md](services/) and
  [libs/*/README.md](libs/) — required env vars, local run command,
  test invocations.
- **[deploy/README.md](deploy/README.md)** — compose-file layout,
  devcontainer setup, prod host bootstrap.

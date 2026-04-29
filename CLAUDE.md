# fishsense-lite-mono — Claude operating notes

Loose ends and architectural conventions that aren't otherwise tracked.

## Service map

| Service | Purpose | Task queue |
|---|---|---|
| `services/fishsense-api/` | FastAPI app (DB CRUD, label endpoints) | — |
| `services/fishsense-api-workflow-worker/` | api-side Temporal worker (NAS + Label Studio orchestration; no activities yet) | (TBD) |
| `services/fishsense-data-processing-workflow-worker/` | image preprocessing (rectify/overlay/JPEG) | `fishsense_data_processing_queue` |
| `services/fishsense-backup-worker/` | nightly Postgres → NAS backups + retention | `fishsense_backup_queue` |

The backup worker is **deliberately separate** from the data-processing
worker so it doesn't have to share Postgres credentials. It's the only
service in the repo that runs `pg_dump` and the only one with a
`postgres.*` config section. Its blast radius is "read every DB it has
creds for + write to a single NAS root" — narrower than mixing it into
a worker that already does heavy image processing.

The schedule is registered idempotently at worker startup: first
deploy creates `fishsense-daily-db-backup` (cron `0 3 * * *` UTC),
subsequent deploys see "already exists" and leave it alone. To change
config (cron, retention, db list), an operator must
`temporal schedule delete fishsense-daily-db-backup` and let the next
worker startup recreate it — refusing to update in-place avoids a
config typo silently retiring the schedule.

Default DB list: `fishsense`, `superset`, `temporal_db`. Skipping
`temporal_visibility_db` (rebuildable index) and the system `postgres`
DB (negligible). To add or remove, override
`E4EFS_BACKUP__DATABASES='["a","b"]'` (or set in settings.toml).

## Notebook port status

| Stage | Notebook | Owner | Status |
|---|---|---|---|
| 0.1 | preprocess_laser_images | data-worker | ported |
| 0.3 | populate_label_studio_project | api-worker | not started |
| 1   | cluster_dive_frames | data-worker | ported (pre-existing) |
| 2   | preprocess_dive_images | data-worker | ported |
| 4   | populate_label_studio_project | api-worker | not started |
| 4.2 | sync_species_labels | api-worker | not started |
| 5.1 | preprocess_headtail_images | data-worker | ported |
| 5.3 | populate_label_studio_project | api-worker | not started |
| 6.1 | update_dive_image_groups | api-worker | not started |
| 9   | preprocess_slate_images | data-worker | ported |
| 11  | populate_label_studio_project | api-worker | not started |
| 12  | sync_slate_label | api-worker | not started |
| 13  | perform_laser_calibration | api-worker | not started (kernel in fishsense-core) |
| 14  | measure_fish | api-worker | not started (kernel in fishsense-core) |

Owner column is the *target* worker once the api-worker side is built;
currently the api-worker has no activities yet, so all api-worker
notebooks are still hand-run.

## Data-worker activity pattern

Every ported per-image stage follows the same shape:

1. `download_raw(checksum)` from the file-exchange.
2. (stage 9 only) `download_slate_pdf(slate_id)`.
3. Off-loop CPU work via `asyncio.to_thread`: rectify
   (`RectifiedImage(RawImage(bytes), intrinsics)` — rawpy + auto-gamma +
   CLAHE + `cv2.undistort`) → stage-specific overlay → `cv2.imencode`.
4. `upload_processed_jpeg(folder, checksum, jpeg_bytes)` to the
   file-exchange.

Output folders match the labeler-facing GET routes already in
`deploy/static_file_server/nginx.conf`:

| Stage | Folder |
|---|---|
| 2   | `preprocess_groups_jpeg` |
| 0.1 | `preprocess_jpeg` |
| 5.1 | `preprocess_headtail_jpeg` |
| 9   | `preprocess_slate_images_jpeg` |

Each port has the same 4-test TDD structure: pure-logic overlay/encode
unit tests, in-process Temporal workflow contract test, integration
test against real `.ORF` fixture (`-m integration`), notebook byte-parity
test (`-m integration`). The integration + parity tests share the same
`tests/fixtures/stage2_sample.ORF` — there's no per-stage raw fixture.

Stage 5.1's parity test also doubles as a proof that
`RawImage(p).data → cv2.undistort(...)` equals
`RectifiedImage(RawImage(p), intrinsics).data` byte-for-byte.

The stages are *intentionally not refactored* into a shared base
activity. Each has a distinct overlay shape (text vs rectangle vs
PDF-composite) and a distinct DTO; one shared signature would have to
be `Callable[[ndarray], ndarray]` plus union-typed extra args, which
is messier than four small, self-contained activities.

## New workflows are gated by `feature_flags.new_preprocess_workflows`

The four ports above (stages 0.1, 2, 5.1, 9) are registered with the
production worker only when
`E4EFS_FEATURE_FLAGS__NEW_PREPROCESS_WORKFLOWS=true`. Default is OFF —
deploying the binary without that env set means only the legacy
`DiveFrameClusteringWorkflow` runs. Calls to `start_workflow` for the
new types still succeed server-side but the workflow tasks sit in the
queue forever (no worker claims them), so the gate is effectively a
soft block, not a hard error.

Lift the flag once the api-worker driver for a given stage exists *and*
the relevant math has been re-verified on real frames (especially the
stage 14 sign concern — see project memory). The flag is a single
all-or-nothing switch by design; if you need finer control, split into
per-stage flags rather than gating inside workflow code (workflow code
must stay deterministic).

Integration tests pass workflows directly to a one-off `Worker(...)`
and don't depend on the registration gate, so the flag has no effect
on tests.

## File-exchange URL contract

```
GET  /api/v1/exchange/raw/{checksum}.ORF             # api-worker stages, data-worker reads
GET  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf # api-worker stages, data-worker reads (stage 9)
PUT  /api/v1/exchange/{folder}/{checksum}.JPG        # data-worker writes
```

The nginx DAV alias at `/api/v1/exchange/` covers any subpath, so
adding new conventions is a `FileExchangeClient` change only — no
nginx.conf change needed.

## Worker config validation gotcha

Dynaconf eagerly validates **every** `Validator` on first attribute
access of `settings`, not lazily per setting. Tests that import any
activity module must plumb env values for all required settings
(`temporal.host`, `e4e_nas.url`, `fishsense_api.url`, etc.) even if
the test only uses one of them — see `configure_worker_settings` in
`test_stage2_integration.py` for the standard placeholder fixture.

The `*.url` validators use a custom `_url_condition` (http/https +
non-empty hostname) instead of `validators.url`, because the strict
library condition rejects every Docker-internal hostname
(`static_file_server`, `fishsense-api`, `temporal` — underscores or no
TLD). Don't switch back to `validators.url`.

## Repo-root `settings.toml` — do NOT commit

`fishsense_shared.get_config_path()` falls back to `cwd` outside Docker,
so the worker reads `./settings.toml` when run from the repo root.
Running it locally creates this file as a side-effect; it has prod-y
URLs inside.

Polyrepo `settings.toml` leftovers were intentionally cleaned up in
`6c3920b` and the same file coming back from local-running is the same
problem. Leave it untracked. If a committed file is genuinely needed,
the right shape is `settings.toml.example` + gitignore (matches the
`deploy/.env.local.example` pattern).

## `fishsense-core` 1.7.0 → 1.7.1 was bundled into the stage-2 port

Bumped in working tree before the stage-2 commit and rolled into commit
`669f933` rather than split out. If you're tracing why a particular
fishsense-core version is in `uv.lock`, look at `669f933` *and* the
workspace pyproject change in `75d2979` (the prior 1.7.0 bump).

## CI pipeline: build → release → promote → deploy

The four-workflow pipeline:

```
push to main         → build.yml    (image -> :sha-<short> + :main)
release-please merge → release.yml  (cuts GitHub release + tag)
release: published   → promote.yml  (:sha-<short> -> :v<version> + :latest;
                                      opens auto-deploy/* PR bumping
                                      deploy/compose*.yml pin)
auto-deploy PR merge → deploy.yml   (docker compose pull && up -d on
                                      self-hosted [fishsense-prod] runner)
```

`.github/workflows/build.yml` runs on **every push to main + every PR**.
On push to main it pushes to GHCR tagged by the commit SHA
(`:sha-<short>`) and the branch (`:main`). PR runs build only — no
push — as a Dockerfile-validity check.

`.github/workflows/promote.yml` runs on `release: published` (fired by
release-please after the release PR is merged). It does **not**
rebuild — (a) retags the SHA-tagged image to `:v<version>` and
`:latest` via `docker buildx imagetools create` (manifest-only push,
no layer transfer); (b) opens a PR on `deploy/compose*.yml` bumping
the package's image pin to the new version. Branch name pattern:
`auto-deploy/<package>-<version>`.

`.github/workflows/deploy.yml` runs when an `auto-deploy/*` PR is
merged (via `pull_request: types:[closed]` + branch-prefix filter, so
unrelated compose edits don't trigger it). Plus a `workflow_dispatch`
for manual re-pulls. Runs `docker compose pull && up -d` on a
self-hosted runner labeled `fishsense-prod`, co-located with the
docker engine running `deploy/compose.yml`. **The runner doesn't
exist yet** — until one is registered, deploy jobs sit in queue.

The workflow operates on a **persistent ops-managed deploy directory**
on the host (path supplied via repo variable `DEPLOY_DIR`),
NOT the runner's default `_work` checkout. This matters because
`deploy/compose*.yml` uses relative bind mounts (`./pg_volumes`,
`./worker_volumes`, `./.secrets/...`) for postgres data, worker
config, postgres admin password, temporal env files, etc. — none of
which are tracked in git. Running compose against the runner's
fresh `_work` checkout would silently start postgres with an empty
data dir.

Host bootstrap (one-time):
1. Register the runner with `--labels fishsense-prod`.
2. `git clone` the repo to a persistent path (e.g. `/srv/fishsense`).
3. Set repo variable `DEPLOY_DIR` to that path under Settings ->
   Secrets and variables -> Actions -> Variables.
4. Restore `pg_volumes/`, `worker_volumes/`, `temporal_volumes/`,
   `mafl_volumes/`, and `.secrets/` (untracked siblings of the
   compose files) from existing prod state.

Three reasons for the split:
1. **Race-proof promotion.** The release tag points at a specific
   commit SHA. Promote retags the image built from that exact SHA,
   not whatever happens to be `:latest`. If a newer non-release
   commit lands on main between the release-please merge and promote
   running, the wrong image can't get tagged with the release version.
2. **Don't pay the build cost twice.** build.yml already built the
   image when the release commit landed; promote.yml is a manifest
   retag (~seconds).
3. **Intentional deploy.** deploy.yml only fires when a human merges
   the auto-deploy PR. The compose-pin diff is reviewable in the PR
   before any prod restart happens.

`fishsense-data-processing-workflow-worker` is held off auto-deploy:
its image still gets the `:v<version>` tag (so manual rollout is
possible), but no compose-pin PR is opened. The data-worker runs on
a separate host and that host's compose isn't in this repo yet.

`deploy/compose.workers.yml` is the home for `fishsense-*` worker
services running on the orchestrator host. Currently has
`fishsense-api-workflow-worker` (moved out of `compose.temporal.yml`
on 2026-04-29 — workers consume Temporal but aren't part of the
cluster). `fishsense-backup-worker`'s stanza will land here once the
prod `backup` Postgres role + NAS creds are set up.

Race guard: promote.yml polls for the `:sha-<short>` image to appear
(up to 20 min) before retagging. build.yml is triggered by the same
push event and runs in parallel with release-please, so promote may
arrive first.

### Service Dockerfile pattern (monorepo-aware)

All four service Dockerfiles use the same shape — see migration
finding #4 in project memory for the rationale and prior broken state.

- Build context = repo root (`docker build -f services/<svc>/Dockerfile .`).
- COPY `pyproject.toml uv.lock` + every workspace member's
  `pyproject.toml` (uv requires all of them to satisfy
  `[tool.uv.workspace] members`).
- COPY the source trees the target needs: always its own, plus
  `libs/fishsense-shared` (and `libs/fishsense-api-sdk` for services
  that import it at runtime — currently api-workflow-worker and
  data-processing-workflow-worker; fishsense-api uses it dev-only and
  fishsense-backup-worker doesn't use it at all).
- Single `uv sync --frozen --no-dev --no-editable --package <name>`
  to install runtime deps + the package itself.
- Two-stage build with `python:3.13-slim-trixie` runtime; copy `.venv`
  from builder.
- System libs per-service: opencv-python needs `libgl1 + libglib2.0-0`
  in the data-worker image; backup-worker needs `postgresql-client`.

## release-please bootstrap-sha — bump when the Release job times out

The Release workflow runs `googleapis/release-please-action@v4`, which
walks main's commit history backwards looking for each package's last
release. Because no GitHub releases or tags exist in this monorepo yet
(the manifest versions are inherited from the polyrepos), every run is
a "first release" walk and the GraphQL API has timed out twice with
`We couldn't respond to your request in time` after ~250 commits.

Fix: every package in `release-please-config.json` carries a
`bootstrap-sha` that caps the walk. When the Release job times out
again, bump the `bootstrap-sha` forward to a more recent commit on
main (the last successful Release run, or any recent commit that
predates the unreleased work). Same SHA across all packages is fine —
release-please just needs *some* lower bound to stop paginating.

After the first successful release-please run cuts a release PR + tag,
this becomes self-maintaining (release-please uses its own tags as the
walk floor) and bootstrap-sha can stay pinned forever or be removed.

## Other open work lives in project memory

Migration findings #1 (core↔sdk dep direction) and #4 (service
Dockerfiles broken in monorepo layout), the four `label_studio_json`
SDK drift allowlist entries, the stage14 sign flip, and the phase-6
cutover items are all in
`~/.claude/projects/.../memory/`. Start there for status of in-flight
work.

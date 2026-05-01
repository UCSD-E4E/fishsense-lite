# fishsense-lite-mono — Claude operating notes

Loose ends and architectural conventions that aren't otherwise tracked.

## Service map

| Service | Purpose | Task queue |
|---|---|---|
| `services/fishsense-api/` | FastAPI app (DB CRUD, label endpoints) | — |
| `services/fishsense-api-workflow-worker/` | api-side Temporal worker: hourly Label Studio sync (laser/headtail), Superset dashboard-config writer, on-demand Create/Populate × {Laser,Species,HeadTail,DiveSlate} LS project workflows | `fishsense_api_queue` |
| `services/fishsense-data-processing-workflow-worker/` | image preprocessing (rectify/overlay/JPEG), laser calibration, fish measurement | `fishsense_data_processing_queue` |
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
| 0.3 | populate_label_studio_project | api-worker | ported |
| 1   | cluster_dive_frames | data-worker | ported (pre-existing) |
| 2   | preprocess_dive_images | data-worker | ported |
| 4   | populate_label_studio_project | api-worker | ported |
| 4.2 | sync_species_labels | api-worker | partial (sync runs as `SyncLabelStudio*LabelsWorkflow`; species-specific sync TBD) |
| 5.1 | preprocess_headtail_images | data-worker | ported |
| 5.3 | populate_label_studio_project | api-worker | ported |
| 6.1 | update_dive_image_groups | api-worker | not started |
| 9   | preprocess_slate_images | data-worker | ported |
| 11  | populate_label_studio_project | api-worker | ported |
| 12  | sync_slate_label | api-worker | not started |
| 13  | perform_laser_calibration | data-worker | ported (kernel in fishsense-core) |
| 14  | measure_fish | data-worker | ported (kernel in fishsense-core) |

Create and populate are split into separate workflows per stage:

* **`Create<Stage>LabelStudioProjectWorkflow()`** — calls
  `create_<stage>_label_studio_project_activity` which idempotently
  finds the LS project by title (`FishSense — <Stage> Labeling`) or
  creates it from a stored labeling-config XML constant. Returns
  `project_id`. The XML constants in
  `activities/create_*_label_studio_project_activity.py` are empty
  placeholders by default — paste from the existing prod LS project
  (Project Settings -> Labeling Interface -> Code) before relying on
  the create branch. With existing prod projects already in place,
  re-running the create workflow just returns their IDs.
* **`Populate<Stage>LabelStudioProjectWorkflow(dive_id)`** — calls
  `get_active_<stage>_label_studio_project_ids_activity` (SDK query
  for projects with at least one incomplete label of this kind), then
  fans out `populate_<stage>_label_studio_project_activity(dive_id,
  project_id)` across the returned set with a workflow-level
  `Semaphore(4)`. No config IDs — populate's target set is computed
  from SQL. Empty result is a no-op.

Both are registered but not scheduled — they're on-demand
(`temporal workflow start` with a `dive_id` for populate, no args for
create). The eight workflows are: Create/Populate × Laser/Species/
HeadTail/DiveSlate.

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
cluster) and `fishsense-backup-worker`. The backup worker reads its
postgres + NAS credentials from `./backup_worker_volumes/config/`
(`settings.toml` + `.secrets.toml`); that directory must be
populated on the host before the service will start successfully.

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

## Operational ground truth (read before touching prod)

### No staging / test environment

`orchestrator.fishsense.e4e.ucsd.edu` is production *and* the only
deployed instance — this is a research codebase, a parallel staging
tier was never built. The rollback mechanism is the nightly DB backup
written by `fishsense-backup-worker`.

Implications when changing prod-touching code:
- Any path that writes to fishsense-api (POST/PUT/DELETE) deserves
  per-cohort confirmation before running against prod, plus a
  read-only dry run first.
- When validating refactored math against legacy outputs, prefer
  running new code in **read-only mode against dives that already
  have computed values in prod** — pre-refactor `Measurement` rows are
  a free oracle. Compare numerically before writing.
- Don't ask "test vs prod"; that question has no answer here. Ask
  about safety gates instead (commented-out writes, sample-size caps,
  human review before persistence).

### Authentik fronts the public API — SDK basic auth gets 302'd

`orchestrator.fishsense.e4e.ucsd.edu` is fronted by Traefik with the
`authentik@docker` middleware (see
[deploy/compose.orchestrator.yml](deploy/compose.orchestrator.yml)).
`fishsense-api-sdk.Client` uses HTTP Basic auth, so a request from a
dev box gets a 302 redirect to authentik's OAuth flow and
`raise_for_status()` blows up — even with valid credentials. Workers
running on the orchestrator host hit fishsense-api on the internal
docker network and skip the proxy entirely.

For dev access, in rough order of effort:
1. ssh / port-forward into the orchestrator's docker network and
   point `fishsense_api.url` at the internal address.
2. Bypass the API and read Postgres directly (host is in
   `deploy/fishsense_api_volumes/config/settings.toml`).
3. Have ops add an authentik basic-auth-passthrough policy for the
   service user on the relevant API paths.
4. Modify the SDK to do `client_credentials` against authentik.

For read-only validation work, option 2 is cleanest: no auth headache
and no API write surface to worry about.

### `E4EFS_` envvar prefix everywhere

Every service uses `Dynaconf(envvar_prefix="E4EFS", ...)`. The
data-processing worker previously used `DYNACONF_` in the polyrepo —
**deploy hosts running the older worker still have `DYNACONF_*`
variables and must rename them before redeploying this version**.
Dynaconf will silently fail to pick up the old prefix.

### SDK ↔ API model mirror — drift caught by CI

The SQLModel ORM in
[services/fishsense-api/src/fishsense_api/models/](services/fishsense-api/src/fishsense_api/models/)
is the source of truth. Pydantic mirrors in
[libs/fishsense-api-sdk/src/fishsense_api_sdk/models/](libs/fishsense-api-sdk/src/fishsense_api_sdk/models/)
are kept in sync **by hand** — same field name + same type. Drift is
caught by [services/fishsense-api/tests/test_sdk_drift.py](services/fishsense-api/tests/test_sdk_drift.py),
which parametrizes every paired model and asserts (a) field-name
parity and (b) structural type parity (modulo cross-module enum
identity).

Intentional differences (encoded in the test):
- `_CameraIntrinsics` (SDK wire) ↔ `CameraIntrinsics` (API SQLModel) —
  SDK keeps an ergonomic numpy wrapper.
- `_LaserExtrinsics` ↔ `LaserExtrinsics` — same pattern.
- `DiveFrameCluster` (SDK) ↔ `DiveFrameClusterJson` (API) — API splits
  persistence + JSON.

Known allowlisted drift: `label_studio_json` on DiveSlateLabel /
HeadTailLabel / LaserLabel / SpeciesLabel — SDK accepts
`Dict[str, Any] | str | None`, API accepts `Dict[str, Any] | None`.
Reconciling needs a product call (does the SDK round-trip stringified
payloads?). The allowlist has hygiene checks: if a drift is fixed,
the test fails until the allowlist entry is dropped.

When you change a SQLModel field, mirror it in the SDK in the same PR
or update `KNOWN_FIELD_DRIFT` with reasoning.

## Open follow-ups

These don't show up by reading the code; they need explicit tracking.

### Migration finding #1 — `fishsense-core` → `fishsense-api-sdk` dep is backwards

`fishsense-core` only imports `CameraIntrinsics` from the SDK in
`fishsense_core/image/rectified_image.py`. A compute lib should not
depend on an HTTP CRUD client. Fix: relocate `CameraIntrinsics` (and
the rest of intrinsics/extrinsics types) into core, or into a
shared types-only package. Requires coordinated releases of core +
sdk + bumping core's git ref in the data-processing worker.

This also gates removing the
`override-dependencies = ["fishsense-api-sdk"]` workaround in the
workspace root `pyproject.toml` — the SDK was folded into the
workspace 2026-04-27 and the override is what forces the workspace
path to win over fishsense-core's transitive git source for the SDK.

### Phase 6 polyrepo cutover leftovers

The four old polyrepos — `fishsense-api`, `fishsense-api-workflow-worker`,
`fishsense-data-processing-workflow-worker`, `fishsense-web-services` —
are still **not archived** on GitHub as of 2026-05-01. Pending:

- Add a `MIGRATED_TO_MONOREPO.md` notice to each, push, then archive.
- `fishsense-core` has a local commit adding `WorldPointHandler`
  PyO3 bindings that data-processing notebooks now depend on. Push,
  let release-please cut a minor, then bump the core git ref in
  `services/fishsense-data-processing-workflow-worker/pyproject.toml`.
- Update Slack pinned messages, lab wiki entries, and any external
  docs that reference old polyrepo URLs.

These are coordination tasks across GitHub repos and external
systems — confirm with the user before pushing or archiving anything.

### stage14 sign-flip — math layer verified, real-frame regression still open

The notebook refactors in
`services/fishsense-data-processing-workflow-worker/scripts/`
(stage13/14/5.1) delegate previously-inline algorithms (atanasov
calibration, K^-1 projection, laser triangulation, raw decoding) to
`fishsense_core`. The concern was that stage14's
`compute_world_point_from_depth` was rewritten to call
`WorldPointHandler.compute_world_point_from_depth` and dropped the
notebook's external `* -1` sign flip — `compute_world_point_from_laser`
feeds depth into `_from_depth` via `laser3d[2]` and might disagree on
sign convention.

Synthetic-geometry tests (commits `15a545a`, `a5e92c6`) pin down the
math:
- `tests/test_compute_world_point_from_depth_convention.py` — kernel
  is `K^-1 · [x,y,1] · depth` with positive sign, no internal flip.
- `tests/test_stage14_pipeline_sign_consistency.py` — runs stage 14's
  exact handoff sequence on synthetic 3D scenes (on-axis laser,
  off-axis laser, realistic Olympus intrinsics) and asserts recovered
  absolute head/tail positions equal ground truth.

Stage 5.1 was independently verified byte-for-byte by
`tests/test_stage5_1_notebook_parity.py`.

**Still open**: real-frame regression. None of the stage 13/14
refactors have been re-run against a dive with historically-recorded
`Measurement` rows in the DB. That needs api-worker + prod-DB access;
once available, a few-pixel comparison vs old measurements is the
gold-standard verification. Don't claim "fully verified" until that
runs.

### Label Studio create-then-populate bootstrap

Eight workflows: Create + Populate × {Laser, Species, HeadTail,
DiveSlate}. Populate's target set is queried via SDK
`get_<stage>_label_studio_project_ids(incomplete=True)` — projects
with at least one not-yet-completed label.

**Bootstrap chicken-and-egg:** a freshly-created project from the
Create workflow has zero labels and won't be returned by the populate
query until something seeds it. Existing prod projects (laser=73,
species=70, headtail=71, dive_slate=66 as of 2026-01-26) already
have labels, so populate finds them. For brand-new deployments this
gap will need to be closed — likely by Create pushing a sentinel
initial label, or by composing Create + Populate into a parent
workflow. Surface this when rolling out a fresh stage.

The four `<STAGE>_LABELING_CONFIG_XML` constants in
`activities/create_*_label_studio_project_activity.py` are empty
placeholders — paste from the existing prod LS project (Project
Settings → Labeling Interface → Code) before relying on the create
branch in a fresh deployment.

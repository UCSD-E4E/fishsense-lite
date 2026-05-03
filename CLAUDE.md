# fishsense-lite-mono — Claude operating notes

Loose ends and architectural conventions that aren't otherwise tracked.

## Service map

| Service | Purpose | Task queue |
|---|---|---|
| `services/fishsense-api/` | FastAPI app (DB CRUD, label endpoints) | — |
| `services/fishsense-api-workflow-worker/` | api-side Temporal worker: hourly Label Studio sync (laser/headtail/dive-slate/species), Superset dashboard-config writer, on-demand Create/Populate × {Laser,Species,HeadTail,DiveSlate} LS project workflows, hourly preprocess parents for stages 0.1 / 2 / 5.1 / 9 (select + resolve; dispatch child to data-worker) | `fishsense_api_queue` |
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
| 0.1 | preprocess_laser_images | api-worker (parent) + data-worker (child) | ported (hourly) |
| 0.3 | populate_label_studio_project | api-worker | ported |
| 1   | cluster_dive_frames | data-worker | ported (pre-existing; no api-worker parent yet — workflow doesn't write back to DB) |
| 2   | preprocess_dive_images | api-worker (parent) + data-worker (child) | ported (hourly, +15min offset) |
| 4   | populate_label_studio_project | api-worker | ported |
| 4.2 | sync_species_labels | api-worker | ported (hourly) |
| 5.1 | preprocess_headtail_images | api-worker (parent) + data-worker (child) | ported (hourly, +30min offset) |
| 5.3 | populate_label_studio_project | api-worker | ported |
| 6.1 | update_dive_image_groups | api-worker | ported (on-demand) |
| 9   | preprocess_slate_images | api-worker (parent) + data-worker (child) | ported (hourly, +45min offset) |
| 11  | populate_label_studio_project | api-worker | ported |
| 12  | sync_slate_label | api-worker | ported (hourly) |
| 13  | perform_laser_calibration | api-worker (parent) + data-worker (child) | ported (hourly, +50min offset) |
| 14  | measure_fish | api-worker (parent) + data-worker (child) | ported (on-demand; idempotency caveat — see notes) |

Create and populate are split into separate workflows per stage:

* **`Create<Stage>LabelStudioProjectWorkflow()`** — calls
  `create_<stage>_label_studio_project_activity` which idempotently
  finds the LS project by title (`FishSense — <Stage> Labeling`) or
  creates it from a stored labeling-config XML constant. Returns
  `project_id`. The four `<STAGE>_LABELING_CONFIG_XML` constants in
  `activities/create_*_label_studio_project_activity.py` are real
  pasted-from-prod XML (laser, species, headtail, dive-slate) — re-
  running create against an existing prod LS project just returns
  the existing ID via title match. The Create workflow can be invoked
  on-demand, but `Populate<Stage>LabelStudioProjectWorkflow` also
  calls the Create activity internally (see below) so a manual
  Create run is rarely needed.
* **`Populate<Stage>LabelStudioProjectWorkflow(dive_id)`** — calls
  `create_<stage>_label_studio_project_activity` first to ensure the
  canonical project exists (idempotent), then
  `get_active_<stage>_label_studio_project_ids_activity` (SDK query
  for projects with at least one incomplete label of this kind), then
  fans out `populate_<stage>_label_studio_project_activity(dive_id,
  project_id)` across the **union** of the canonical project ID and
  the discovery result, deduplicated, with a workflow-level
  `Semaphore(4)`. The Create-then-discovery union closes the
  bootstrap chicken-and-egg: a freshly-created project has zero
  incomplete labels, so the discovery query alone wouldn't pick it
  up — including its ID in the fan-out set seeds the first round of
  `LaserLabel` / `SpeciesLabel` / `HeadTailLabel` / `DiveSlateLabel`
  rows. Steady-state: discovery still picks up legacy/additional
  projects (e.g. the prod laser project from before Create's XML
  was checked in).

Both Create and Populate are registered but not scheduled — they're
on-demand (`temporal workflow start` with a `dive_id` for populate,
no args for create). The eight workflows are: Create/Populate ×
Laser/Species/HeadTail/DiveSlate.

The four populate workflows are also dispatched automatically as
child workflows from the matching preprocess parent (stages 0.1 →
laser, 2 → species, 5.1 → headtail, 9 → dive-slate). After
`archive_processed_jpegs_to_nas_activity` and
`cleanup_raw_bytes_for_dive_activity`, the parent runs
`execute_child_workflow("Populate<Stage>LabelStudioProjectWorkflow",
dive_id, id="populate-<stage>-{dive_id}",
id_reuse_policy=ALLOW_DUPLICATE_FAILED_ONLY)`. Steady-state behavior
on the same cohort dive: hour 1 imports tasks; hours 2+ catch
`WorkflowAlreadyStartedError` and no-op rather than re-importing
duplicate LS tasks. On-demand invocation is still supported for
backfill or operator intervention; if you trigger it manually, use a
non-colliding workflow id (e.g. `populate-laser-393-manual`) so the
auto-chain's deterministic id stays available for future hourly
firings.

`UpdateDiveImageGroupsWorkflow(dive_id)` is the stage-6.1 on-demand
workflow: it walks the dive's PREDICTION clusters, looks up each
entry's `SpeciesLabel.grouping`, and POSTs LABEL_STUDIO clusters
according to the labelers' "Part of previous group" / "Not part of
current group" choices. Stage 14 measurement reads those clusters.
The activity refuses to re-run when LABEL_STUDIO clusters already
exist — the cluster API has no DELETE, so a re-POST would silently
double-count. To re-group after labels change, an operator must
manually drop the existing LABEL_STUDIO clusters first.

## Cross-worker orchestration pattern (stages 0.1, 2, 5.1, 9, 13, 14)

The api-worker is the brains; the data-worker is the executor. Stages
that need both SDK-side decision-making *and* CPU-heavy per-image
work split into two workflows:

* **Parent** on api-worker (`fishsense_api_queue`). Hourly schedule.
  Activity calls per dive, bracketing the data-worker child plus the
  in-process LS-populate child:
  1. Selector — returns next dive_id in cohort, or None.
  2. Resolver — returns a fully-populated workflow-input DTO.
  3. `stage_raw_bytes_for_dive_activity` — NAS → file-exchange.
     Stage 9 also runs `stage_slate_pdf_activity` for the slate PDF.
  4. `start_child_workflow` against the data-worker task queue.
  5. `archive_processed_jpegs_to_nas_activity` — file-exchange JPEGs
     → NAS (`processed_jpegs/<workflow>/<dive_id>/<checksum>.JPG`).
     Then `cleanup_raw_bytes_for_dive_activity` deletes the dive's
     raw `.ORF`s from the file-exchange.
  6. `execute_child_workflow("Populate<Stage>LabelStudioProjectWorkflow",
     dive_id, id="populate-<stage>-{dive_id}",
     id_reuse_policy=ALLOW_DUPLICATE_FAILED_ONLY)` — on-demand
     populate child runs against the same task queue. The reuse
     policy + deterministic id deduplicate against re-firings of the
     parent on the same cohort dive — once labels start completing,
     the dive drops out of the cohort, but if the parent fires twice
     on the same dive_id (for whatever reason) the second populate
     hits `WorkflowAlreadyStartedError` and the parent catches it so
     the post-archive run still completes successfully.
* **Child** on data-worker (`fishsense_data_processing_queue`). Thin
  pre-input workflow that fans out per-image activities. No SDK
  calls and no NAS calls; all bytes already on the file-exchange,
  all decisions baked into the input DTO.

NAS access lives only on the api-worker side. The data-worker stays
file-exchange-only — narrows the data-worker's blast radius and
keeps NAS credentials off the cluster.

JPEGs intentionally stay on the file-exchange after archive (LS task
URLs point at them). Raw `.ORF`s are deleted because they're
reproducible from NAS. JPEG retention is a separate operational
decision — see the project memory entry.

The workflow-input DTOs (`PreprocessLaserImagesInput`, eventually
`PreprocessDiveImagesInput`, etc.) live in
[fishsense_shared.preprocess_contracts](libs/fishsense-shared/src/fishsense_shared/preprocess_contracts.py)
because they're the api-worker / data-worker contract; per-image
DTOs stay in the data-worker workflow modules because they're
internal to the fan-out.

**Cluster correctness** (relevant once the data-worker scales beyond
one replica):

* Schedule for the parent uses
  `overlap=ScheduleOverlapPolicy.SKIP` — a previous run still in
  flight blocks the next firing, so two selectors can't race past
  the same `dives.get()` and pick the same dive.
* Child workflow id is deterministic (`preprocess-laser-{dive_id}`)
  and dispatched with
  `id_reuse_policy=ALLOW_DUPLICATE_FAILED_ONLY`. If a parent run
  *does* race past the schedule guard (e.g. manual trigger
  overlapping a scheduled one), the second `start_child_workflow`
  raises `WorkflowAlreadyStartedError` which the parent catches —
  archive + cleanup + populate then still run, so a child-then-parent
  split failure self-heals on the next firing without redoing
  per-image work. **Note:** temporalio's default child
  `id_reuse_policy` is `ALLOW_DUPLICATE`, which lets duplicates
  through silently — the explicit setting is required.
* Per-image activities are idempotent: nginx DAV PUT overwrites,
  SDK upserts.

Applied to stages 0.1, 2, 5.1, 9, 13 — each parent runs hourly. The
four preprocess parents are slotted at +0/+15/+30/+45 min so their
selectors don't all hit `dives.get()` at the top of the hour; stage
13 calibration sits at +50 min. Stage 14 measurement is registered
but not scheduled (see notes below). Per-stage cohort:

| Stage | Parent cohort definition |
|---|---|
| 0.1 | HIGH-priority + at least one image without ANY `LaserLabel` row (in any project) |
| 2   | HIGH-priority + has PREDICTION clusters + at least one image without ANY `SpeciesLabel` row |
| 5.1 | HIGH-priority + at least one `SpeciesLabel.top_three_photos_of_group=True` whose image carries no `HeadTailLabel` row at all |
| 9   | HIGH-priority + `dive_slate_id` set + at least one `SpeciesLabel.content_of_image='Slate, Laser on slate'` whose image carries no `DiveSlateLabel` row at all |
| 13  | HIGH-priority + `dive_slate_id` set + no `LaserExtrinsics` + ≥2 completed `DiveSlateLabel` rows (matches the data-worker activity's `MIN_LASER_POINTS=2` precondition) |
| 14  | HIGH-priority + has `LaserExtrinsics` + has LABEL_STUDIO clusters with at least one `fish_id is None` |

The four preprocess cohorts (0.1, 2, 5.1, 9) check "no row at all"
rather than "no completed row" so a dive drops out the moment
populate seeds even-incomplete sentinel rows for every image. The
earlier `completed`-only predicate kept dives in the cohort
indefinitely between populate and labelers finishing — every hourly
firing re-staged raw `.ORF`s from NAS, re-rectified, and re-archived
(child-workflow `ALLOW_DUPLICATE_FAILED_ONLY` made the per-image
work a no-op, but the NAS staging activity ran unconditionally on
every parent firing). Resolver activities mirror the same predicate:
`resolve_laser/headtail/slate_preprocess_inputs_activity` filter
images on "no label row" so the dispatched per-image work matches
what the cohort selector promised.

Stage 1 (clustering) does NOT yet have a parent — its data-worker
workflow returns clusters but doesn't write them back to the DB, so
adding a parent requires deciding whether the parent persists the
output (via `images.post_cluster`) or whether the workflow itself
gains a write step. Defer until a real consumer needs it.

Stages 13 and 14 are structurally lighter than the four preprocess
parents: pure SDK math, no NAS staging, no file-exchange JPEGs, no
per-image fan-out. Their selector + child-dispatch parents have only
two activity calls (selector → `start_child_workflow`); the data-worker
keeps SDK fetches inline because the math kernels need opencv +
fishsense-core, so splitting fetch/math across workers would add 5+
activity handoffs per dive for no gain.

**Stage 14 deliberately is not scheduled.** `measure_fish_activity` is
non-idempotent (`post_measurement` is a POST and the SDK has no
per-image measurement query), so a re-run on a partially-failed dive
would duplicate measurements on already-bound clusters. Until that's
resolved (likely by adding a `get_measurements` SDK method and
per-image filtering in the activity), `MeasureFishParentWorkflow` is
operator-triggered:

```
temporal workflow start \
    --task-queue fishsense_api_queue \
    --type MeasureFishParentWorkflow \
    --workflow-id measure-fish-parent-<run-tag>
```

Each invocation drains exactly one dive — call it repeatedly to clear
a backlog.

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
                                      the matching self-hosted runner —
                                      [fishsense-prod] for the orchestrator
                                      stack, [fishsense-data-worker] for
                                      the data-processing worker host)
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
for manual re-pulls. Two jobs route by branch name:

- `auto-deploy/fishsense-data-processing-workflow-worker-*` ->
  `[self-hosted, fishsense-data-worker]`, repo variable
  `DATA_WORKER_DEPLOY_DIR`, `compose.data-worker.yml`.
- any other `auto-deploy/*` -> `[self-hosted, fishsense-prod]`, repo
  variable `DEPLOY_DIR`, `compose.yml` (which `include:`s the four
  orchestrator-stack siblings).

**Neither runner exists yet** — until each is registered, the matching
deploy jobs sit in queue.

Each job operates on a **persistent ops-managed deploy directory**
on its host (paths in `DEPLOY_DIR` / `DATA_WORKER_DEPLOY_DIR`), NOT
the runner's default `_work` checkout. This matters because
`deploy/compose*.yml` uses relative bind mounts (`./pg_volumes`,
`./worker_volumes`, `./.secrets/...`, `./temporal_volumes/certs`)
for postgres data, worker config, secrets, and Temporal mTLS certs —
none of which are tracked in git. Running compose against the
runner's fresh `_work` checkout would silently start postgres with
an empty data dir on the orchestrator, or the data-worker without
its mTLS certs.

Host bootstrap (one-time, per host):

Orchestrator:
1. Register a runner with `--labels fishsense-prod`.
2. `git clone` the repo to a persistent path (e.g. `/srv/fishsense`).
3. Set repo variable `DEPLOY_DIR` to that path under Settings ->
   Secrets and variables -> Actions -> Variables.
4. Restore `pg_volumes/`, `worker_volumes/`, `temporal_volumes/`,
   `mafl_volumes/`, and `.secrets/` (untracked siblings of the
   compose files) from existing prod state.

Data-worker:
1. Register a runner with `--labels fishsense-data-worker`.
2. `git clone` the repo to a persistent path (e.g. `/srv/fishsense-data-worker`).
3. Set repo variable `DATA_WORKER_DEPLOY_DIR` to that path.
4. The in-repo `worker_volumes/data_worker/config/settings.toml` is
   the canonical config — flows in via `git pull --ff-only origin main`
   like the api-worker's. Populate `worker_volumes/data_worker/config/.secrets.toml`
   (untracked) and `temporal_volumes/certs/` (a data-worker-specific
   client cert + key + the same root CA) on the host.

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

`fishsense-data-processing-workflow-worker` runs on a separate host
(with its own compose file `deploy/compose.data-worker.yml`, NOT
included by `deploy/compose.yml`) and uses the second deploy.yml job
described above. Its compose-pin PR opens just like the orchestrator
services; the routing in `deploy.yml` sends the merge to the
`fishsense-data-worker` runner instead of `fishsense-prod`.

`deploy/compose.workers.yml` is the home for `fishsense-*` worker
services running on the orchestrator host. Currently has
`fishsense-api-workflow-worker` (moved out of `compose.temporal.yml`
on 2026-04-29 — workers consume Temporal but aren't part of the
cluster) and `fishsense-backup-worker`. The backup worker reads its
postgres + NAS credentials from `./worker_volumes/backup_worker/config/`
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

### Label Studio create-then-populate

Eight workflows: Create + Populate × {Laser, Species, HeadTail,
DiveSlate}. Populate now self-bootstraps: it calls the matching
Create activity inside the workflow body, then unions the canonical
project ID with the SDK
`get_<stage>_label_studio_project_ids(incomplete=True)` discovery
result, deduplicates, and fans out per-project populate activities
across the union. The four `<STAGE>_LABELING_CONFIG_XML` constants
are real pasted-from-prod XML, so Create-on-fresh-deploy stands up a
usable project immediately.

The populate workflows are dispatched automatically by the four
preprocess parents (see "Cross-worker orchestration pattern"); manual
`temporal workflow start` is only needed for backfill of dives the
auto-chain has already cleared, or to recover from a populate that
previously errored out. Use a non-colliding workflow id for manual
runs (e.g. `populate-laser-393-manual`) so the auto-chain's
deterministic id (`populate-laser-393`) stays available for future
hourly firings.

Existing prod projects (laser=73, species=70, headtail=71,
dive_slate=66 as of 2026-01-26) keep being picked up by the
discovery query as long as they hold incomplete labels. The Create
title-match returns the canonical project's ID; if a deployment
ever ends up with the canonical title pointing at a different
project than the one prod is using, both IDs flow through the
union and populate fans out across both — which is the right
behavior during a migration window but not the desired steady
state. Resolve by aligning `<STAGE>_PROJECT_TITLE` in the Create
activity with whatever the operator wants as canonical.

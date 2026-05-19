# fishsense-lite-mono — Claude operating notes

Loose ends and architectural conventions that aren't otherwise tracked.

## Working conventions

**TDD is mandatory** for any non-trivial code change. Write a failing
test first, then the minimum implementation to pass, then refactor.
Applies to: API endpoints, SDK methods, activities, workflows, web app
data/utility modules, and any new business logic. UI rendering is the
narrow exception — manual browser verification is acceptable for
purely visual components, but any logic worth testing should be
extracted into a unit and covered. The data-worker activity ports'
"4-test TDD structure" (pure-logic unit + workflow contract +
integration + parity, see the data-worker activity pattern section)
is the gold standard; smaller modules don't need all four legs but do
need a failing test before the implementation lands.

## Service map

| Service | Purpose | Task queue |
|---|---|---|
| `services/fishsense-api/` | FastAPI app (DB CRUD, label endpoints) | — |
| `services/fishsense-api-workflow-worker/` | api-side Temporal worker: hourly Label Studio sync (laser/headtail/dive-slate/species), on-demand Create/Populate × {Laser,Species,HeadTail,DiveSlate} LS project workflows, hourly preprocess parents for stages 0.1 / 1 / 2 / 5.1 / 9 (select + resolve; dispatch child to data-worker) | `fishsense_api_queue` |
| `apps/fishsense-lite-web/` | Next.js 15 (App Router) + React + TS landing page at `fishsense.e4e.ucsd.edu`. SSR fetches LS project IDs from fishsense-api, resolves names from Label Studio, renders categorized link cards. Auth.js (next-auth v5) with Authentik OIDC gates `/portal/*`; landing stays public. Replaces the prior mafl dashboard + its hourly config-writer workflow. Will grow into a full web app. | — |
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
| 1   | cluster_dive_frames | api-worker (parent) + data-worker (child) | ported (hourly, +5min offset) |
| 2   | preprocess_species_images (renamed from preprocess_dive_images) | api-worker (parent) + data-worker (child) | ported (hourly, +15min offset) |
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

Create and populate are split into separate workflows per stage. LS
projects are now **per-dive**: each dive gets its own LS project
titled `"{dive.name} - <Stage> Labeling"` (e.g. `"2024-08-21 reef
dive 3 - HeadTail Labeling"`), with `f"Dive {dive_id}"` as a
fallback when `Dive.name` is NULL. Per-dive scoping lets labelers
track per-dive progress and keeps each project's task list focused
on one cohort.

* **`Create<Stage>LabelStudioProjectWorkflow(dive_id)`** — calls
  `create_<stage>_label_studio_project_activity(dive_id)` which
  fetches the dive's name from the API, builds the per-dive title
  via `populate_utils.build_per_dive_title`, and idempotently finds
  that title in LS or creates a new project from a stored
  labeling-config XML constant. Returns `project_id`. The four
  `<STAGE>_LABELING_CONFIG_XML` constants in
  `activities/create_*_label_studio_project_activity.py` are real
  pasted-from-prod XML. The Create workflow can be invoked on-demand,
  but `Populate<Stage>LabelStudioProjectWorkflow` also calls the
  Create activity internally so a manual Create run is rarely needed.
* **`Populate<Stage>LabelStudioProjectWorkflow(dive_id)`** — calls
  `create_<stage>_label_studio_project_activity(dive_id)` to
  materialize the per-dive project (idempotent), then runs
  `populate_<stage>_label_studio_project_activity(dive_id,
  project_id)` against that single project. No discovery / fan-out
  — each dive owns one project per stage.

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

## Cross-worker orchestration pattern (stages 0.1, 1, 2, 5.1, 9, 13, 14)

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

The workflow-input DTOs (`PreprocessLaserImagesInput`,
`PreprocessSpeciesImagesInput`, `PreprocessHeadtailImagesInput`,
`PreprocessSlateImagesInput`, `ClusterDiveFramesInput`) live in
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

Applied to stages 0.1, 1, 2, 5.1, 9, 13 — each parent runs hourly.
Schedule slots: 0.1 at +0, 1 at +5, 2 at +15, 5.1 at +30, 9 at +45,
13 at +50 min — staggered so their selectors don't all hit
`dives.get()` at the top of the hour. Stage 14 measurement is
registered but not scheduled (see notes below). Per-stage cohort:

| Stage | Parent cohort definition |
|---|---|
| 0.1 | HIGH-priority + at least one image without ANY `LaserLabel` row (in any project) |
| 1   | HIGH-priority + at least one image with a *valid* `LaserLabel` (`completed=True`, `superseded=False`, `x`/`y` both set) + zero PREDICTION `DiveFrameCluster` rows |
| 2   | HIGH-priority + has PREDICTION clusters + at least one image with a *valid* `LaserLabel` whose image carries no non-sentinel `SpeciesLabel` row |
| 5.1 | HIGH-priority + at least one image with a *valid* `LaserLabel` whose image carries no non-sentinel `HeadTailLabel` row |
| 9   | HIGH-priority + `dive_slate_id` set + at least one `SpeciesLabel.content_of_image='Slate, Laser on slate'` whose image carries no `DiveSlateLabel` row at all |
| 13  | HIGH-priority + `dive_slate_id` set + no `LaserExtrinsics` + ≥2 completed `DiveSlateLabel` rows (matches the data-worker activity's `MIN_LASER_POINTS=2` precondition) |
| 14  | HIGH-priority + has `LaserExtrinsics` + has LABEL_STUDIO clusters with at least one `fish_id is None` |

Stages 1, 2, and 5.1 all cascade from the same "valid laser" gate.
Stage 1 lands PREDICTION clusters that stage 2 then consumes; stage
5.1 has no cluster gate so it fires as soon as a single image's
laser is valid. The `+5/+15` slot pair gives stage 1 a 10-minute
head start on stage 2 — clustering on a ~hundred-image dive
completes in seconds, so a single hourly cycle clears the
laser→clustering→species chain. If stage 1 misses the window for
a particular dive (e.g. fires while the dive's lasers are still
landing), stage 2 picks it up next hour.

The preprocess cohorts (0.1, 2, 5.1, 9) check "no row at all"
rather than "no completed row" so a dive drops out the moment
populate seeds even-incomplete sentinel rows for every image. The
earlier `completed`-only predicate kept dives in the cohort
indefinitely between populate and labelers finishing — every hourly
firing re-staged raw `.ORF`s from NAS, re-rectified, and re-archived
(child-workflow `ALLOW_DUPLICATE_FAILED_ONLY` made the per-image
work a no-op, but the NAS staging activity ran unconditionally on
every parent firing). Resolver activities mirror the same predicate:
`resolve_species/headtail/slate_preprocess_inputs_activity` filter
images on laser-valid + no-non-sentinel-row so the dispatched
per-image work matches what the cohort selector promised.

**Stage 5.1 source flip (2026-05-04).** Head/tail used to cascade
from `SpeciesLabel.top_three_photos_of_group=True`, which forced
labelers through stages 1 → 2 → 4 (cluster → preprocess dive →
species top-3 selection) before any head/tail work could start. As
of 2026-05-04 it cascades from valid laser labels instead: head/tail
preprocess + populate fire as soon as laser labelers + the
validator have signed off on an image. Practical consequences:
head/tail can run in parallel with stages 1/2; head/tail tasks are
created for *every* laser-valid image, not just the species pass's
top-3 per group, so the labeler queue is larger.

**Stage 2 source flip (2026-05-05).** Species labeling used to be
gated only on "image has no species row." Flipped to mirror head/tail:
species fires per-image off valid lasers, in parallel with head/tail,
but keeps the PREDICTION-cluster gate so the data-worker fan-out still
gets cluster context for the "image i of N" overlay. Stage 1 was
promoted from operator-driven to an automated parent at the same time
to keep stage 2's cluster gate satisfied. Renames in this flip:

| Before | After |
|---|---|
| `select_next_for_dive_image_preprocessing` (SDK + endpoint) | `select_next_for_species_preprocessing` |
| `PreprocessDiveImagesInput` | `PreprocessSpeciesImagesInput` |
| `PreprocessDiveImagesParentWorkflow` | `PreprocessSpeciesImagesParentWorkflow` |
| `PreprocessDiveImagesWorkflow` | `PreprocessSpeciesImagesWorkflow` |
| `preprocess_dive_image` activity | `preprocess_species_image` |
| `resolve_dive_image_preprocess_inputs_activity` | `resolve_species_preprocess_inputs_activity` |
| `select_next_high_priority_dive_for_dive_image_preprocessing_activity` | `select_next_high_priority_dive_for_species_preprocessing_activity` |
| schedule id `preprocess-dive-images-workflow-schedule` | `preprocess-species-images-workflow-schedule` |

Operator action at deploy: delete the orphan
`preprocess-dive-images-workflow-schedule` Temporal schedule (the
class it points at no longer exists) and drain or terminate any
in-flight `PreprocessDiveImagesParentWorkflow` runs. The species LS
labeling-config XML was also swapped at this flip — laser keypoints
and the "Slate upside down" choice are gone from the new config; the
species sync activity's laser-keypoint/slate-upside-down extraction
paths are stripped accordingly. New labels write only the still-
present columns; historical species rows keep whatever they had.

## `dive_pipeline_status` view

Postgres view that exposes a wide row per dive with one boolean
column per pipeline stage. Created by alembic migration
`60e82ad5dac7_add_dive_pipeline_status_view`; SQL canonicalized in
[fishsense_api.views](services/fishsense-api/src/fishsense_api/views.py).
Backs Superset dashboards reading the fishsense Postgres connection.

| Column | True iff |
|---|---|
| `dive_id`, `priority`, `dive_slate_id` | identity / dimensions |
| `laser_preprocessed` | every image in the dive has at least one `LaserLabel` row (any project, any state) |
| `laser_labeling_complete` | ≥1 completed-non-superseded `LaserLabel` AND zero incomplete-non-superseded |
| `headtail_preprocessed` | every image carrying a *valid* laser label (completed, not superseded, x/y both set) has a non-sentinel `HeadTailLabel` row |
| `headtail_labeling_complete` | ≥1 completed-non-superseded `HeadTailLabel` AND zero incomplete-non-superseded |
| `has_prediction_clusters` | dive has at least one PREDICTION `DiveFrameCluster` (stage 1 ran and persisted) |
| `dive_images_preprocessed` | `has_prediction_clusters` AND every image carrying a *valid* laser label (completed, not superseded, x/y both set) has a non-sentinel `SpeciesLabel` row |
| `species_labeling_complete` | ≥1 completed `SpeciesLabel` AND zero incomplete (no `superseded` column on this model) |
| `slate_applicable` | `dive_slate_id IS NOT NULL` |
| `slate_preprocessed` | every image with `SpeciesLabel.content_of_image='Slate, Laser on slate'` has a non-sentinel `DiveSlateLabel` row |
| `slate_labeling_complete` | ≥1 completed `DiveSlateLabel` AND zero incomplete |
| `calibrated` | dive has a `LaserExtrinsics` row |
| `measured` | ≥1 LABEL_STUDIO `DiveFrameCluster` AND zero LABEL_STUDIO clusters with `fish_id IS NULL` |

**"Complete" semantics throughout** mirror
`get_dives_with_complete_laser_labeling`: vacuous truth (zero rows of
a kind) reads as `False`, not `True`. A dive with no laser labels at
all is *not* "laser_labeling_complete" — there's nothing to validate.
Same for the other `*_labeling_complete` and `measured` flags.

**Edits** to predicates: change the SQL in `views.py` (single source
of truth — both alembic migration and tests use it), then write a new
alembic revision that drops + recreates the view (Postgres `CREATE OR
REPLACE VIEW` is restrictive about column-shape changes; the
drop/recreate pattern is simpler and the view has no dependents). Add
a test for the new behavior in `test_dive_pipeline_status_view.py`
before changing the SQL.

**Auto-migrate on startup.** `fishsense_api.server.lifespan` runs
`SQLModel.metadata.create_all` first (fresh-env bootstrap — the
alembic baseline is `alter_column`-only and assumes tables already
exist), then `run_alembic_upgrade` (catches up to head, including
non-table artifacts like this view). Both are idempotent against an
existing prod schema; the second is what creates the view on the
deploy that ships its migration. `alembic.ini` does NOT ship in the
runtime image (`uv sync --no-editable` only installs the package
source); `run_alembic_upgrade` builds the Config programmatically with
`script_location` pointed at `<package>/alembic`.

Stage 1 (clustering) was promoted to an automated parent on
2026-05-05. The api-worker's `ClusterDiveFramesParentWorkflow` runs
selector → resolver → data-worker child (`DiveFrameClusteringWorkflow`)
→ persist. The child returns `list[list[int]]` of image_ids per
cluster; the persist activity POSTs one
`DiveFrameCluster(data_source=PREDICTION)` per id-list via
`images.post_cluster`. No NAS staging or file-exchange traffic —
clustering is pure math on `(image_id, taken_datetime)` pairs. The
cohort selector excludes dives that already have *any* PREDICTION
cluster, so this is one-shot per dive; an operator must drop partial
PREDICTION rows manually if a parent run failed mid-persist (the
cohort would otherwise skip the dive forever).

Stages 13 and 14 are structurally lighter than the preprocess parents:
pure SDK math, no NAS staging, no file-exchange JPEGs, no per-image
fan-out. Their selector + child-dispatch parents have only two
activity calls (selector → `start_child_workflow`); the data-worker
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

## Service plumbing gotchas

Four service-layer conventions that fail silently when broken. Note them
before adding controllers, models, alembic migrations, or SDK tests.

### `controllers/__init__.py` is the route registry

Controllers register their routes against the FastAPI `app` singleton
imported from `fishsense_api.server` as a **side effect of being
imported**. [services/fishsense-api/src/fishsense_api/controllers/__init__.py](services/fishsense-api/src/fishsense_api/controllers/__init__.py)
does the side-effect imports — add new controllers there or their routes
will silently not register.

### `database.py` is the model registry

[services/fishsense-api/src/fishsense_api/database.py](services/fishsense-api/src/fishsense_api/database.py)
imports every SQLModel so `SQLModel.metadata.create_all` (called from
`lifespan`) and `alembic --autogenerate` both see them. Forgetting to
import a new model there means it won't appear in autogenerated
migrations and won't be picked up by the fresh-env `create_all` bootstrap.
The `# pylint: disable=unused-import` at the top is intentional — don't
"clean up" those imports.

### Alembic dev workflow needs a valid local config

```bash
uv run alembic revision --autogenerate -m "description"   # new migration
uv run alembic upgrade head                                # apply
```

`alembic.ini` points `script_location` at `src/fishsense_api/alembic`;
its `env.py` imports `pg_connection_string` from `fishsense_api.config`,
so a valid `settings.toml` + `.secrets.toml` (or `E4EFS_*` env vars)
must resolve from cwd before alembic will run. In prod, the runtime
image instead calls `run_alembic_upgrade` programmatically — see
"Auto-migrate on startup" under the dive_pipeline_status view section.

### SDK testing conventions

[libs/fishsense-api-sdk/pyproject.toml](libs/fishsense-api-sdk/pyproject.toml)
sets `asyncio_mode = "auto"` — async tests do **not** need the
`@pytest.mark.asyncio` decorator. All clients inherit `ClientBase`
(httpx + retry on `HTTPStatusError`) and **must be used inside
`async with`** — instantiating a raw client and calling a method
outside the context manager raises `RuntimeError`.

## `E4EFS_DOCKER` toggles config + log roots

[libs/fishsense-shared/src/fishsense_shared/config.py](libs/fishsense-shared/src/fishsense_shared/config.py)
defines `IS_DOCKER` as true only when `E4EFS_DOCKER` is an
**explicitly-truthy string** (`"true"`, `"1"`, `"yes"`, …). Shipped
images set it; in that mode config reads from `/e4efs/config/` and logs
go to `/e4efs/logs/`. Outside Docker, config falls back to cwd (see
"Repo-root `settings.toml` — do NOT commit") and logs go to
`platformdirs.user_log_path`.

Do **not** rewrite as `bool(os.environ.get("E4EFS_DOCKER"))` — any
non-empty string (including `"false"`) would read as Docker mode and
send paths to `/e4efs/*` on a dev box. The current implementation has
an inline comment to that effect; it's a real footgun, not a style
preference.

## Local Temporal

For dev: `temporal server start-dev` + set `temporal.tls = false` in
`settings.toml`. Production uses mTLS with certs under `/certs/` mounted
from `deploy/temporal_volumes/certs` — the same `TLSConfig` is built
from `settings.temporal.{client_cert,client_private_key,server_root_ca_cert,domain}`
across every service that talks to Temporal.

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
                                      opens auto-deploy/* PR bumping the
                                      image pin — deploy/compose*.yml for
                                      orchestrator-stack services, or the
                                      newTag in deploy/k8s/data-worker/
                                      kustomization.yaml for the data-worker)
auto-deploy PR merge → deploy.yml   (orchestrator stack: docker compose
                                      pull && up -d on [fishsense-prod];
                                      data-worker: kubectl apply -k
                                      deploy/k8s/data-worker from a
                                      GitHub-hosted runner, NRP)
```

`.github/workflows/build.yml` runs on **every push to main + every PR**.
On push to main it pushes to GHCR tagged by the commit SHA
(`:sha-<short>`) and the branch (`:main`). PR runs build only — no
push — as a Dockerfile-validity check.

The matrix uses `include:` with explicit `dockerfile` paths per entry
(not derived from the package name) because services live under
`services/` while the web app lives under `apps/`. Adding a new
buildable image means appending one entry to that matrix in build.yml
**and** in `rebuild-from-main.yml` (the recovery counterpart). Adding
a non-Python package additionally needs `release-type: node` (or
similar) on its `release-please-config.json` entry — `apps/fishsense-lite-web` is
the first such consumer; default top-level `release-type: python`
applies to all the rest.

`.github/workflows/promote.yml` runs on `release: published` (fired by
release-please after the release PR is merged). It does **not**
rebuild — (a) retags the SHA-tagged image to `:v<version>` and
`:latest` via `docker buildx imagetools create` (manifest-only push,
no layer transfer); (b) opens an auto-deploy PR bumping the package's
image pin. For most packages that's the `image: ...:v*` pin in
`deploy/compose*.yml`; for `fishsense-data-processing-workflow-worker`
(which runs on NRP/Kubernetes, not compose) it's the `newTag:` in
`deploy/k8s/data-worker/kustomization.yaml`. Branch name pattern:
`auto-deploy/<package>-<version>` either way (deploy.yml routes by
prefix).

**SDK consumer cascade (release.yml `auto-bump-sdk-consumers` job).**
`fishsense-api-sdk` is a workspace dep, not published — workers
consume it via `fishsense-api-sdk = { workspace = true }`, which is
path-based. release-please tracks commits per package path, so an
SDK-only commit doesn't bump the workers, and the workers' compose
pins keep pointing at images built before the SDK fix. The follow-up
job in [release.yml](.github/workflows/release.yml) closes that gap:
when release-please reports `libs/fishsense-api-sdk` in
`paths_released`, it reads the new SDK version from
`.release-please-manifest.json`, updates each consumer's
`"fishsense-api-sdk>=X.Y.Z"` pin in `pyproject.toml`, and opens an
`auto-bump/fishsense-api-sdk-<version>` PR. Merging that PR causes
release-please to bump every consuming worker on the next run, which
then goes through the standard promote → auto-deploy chain.

The list of consumers is hardcoded in the job's `for pkg in ...`
loop. Adding a new SDK consumer means: (a) declaring the
`"fishsense-api-sdk>=X.Y.Z"` pin in its `pyproject.toml`, and (b)
appending its path to the loop. release-please does NOT have a
built-in Python-workspace-aware plugin (unlike `node-workspace` /
`cargo-workspace`); this job is the workaround.

`.github/workflows/deploy.yml` runs when an `auto-deploy/*` PR is
merged (via `pull_request: types:[closed]` + branch-prefix filter, so
unrelated edits don't trigger it). Plus a `workflow_dispatch` for
manual re-deploys. Two jobs route by branch name:

- `auto-deploy/fishsense-data-processing-workflow-worker-*` ->
  `deploy-data-worker` on `ubuntu-latest` (GitHub-hosted; kubectl is
  preinstalled): writes the `NRP_KUBECONFIG` repo secret to a temp
  kubeconfig and runs `kubectl apply -k deploy/k8s/data-worker` +
  `kubectl rollout status`. No persistent ops dir, no docker — the
  kustomization is self-contained and config/certs live in cluster
  ConfigMaps/Secrets.
- any other `auto-deploy/*` -> `deploy-orchestrator` on
  `[self-hosted, fishsense-prod]`, repo variable `DEPLOY_DIR`,
  `compose.yml` (which `include:`s the four orchestrator-stack
  siblings).

**The `fishsense-prod` runner doesn't exist yet** — until it's
registered the orchestrator deploy job sits in queue; the
`deploy-data-worker` job runs but fails fast until `NRP_KUBECONFIG`
is set.

The `deploy-orchestrator` job operates on a **persistent ops-managed
deploy directory** on the host (path in `DEPLOY_DIR`), NOT the
runner's default `_work` checkout, because `deploy/compose*.yml` uses
relative bind mounts (`./pg_volumes`, `./worker_volumes`,
`./.secrets/...`, `./temporal_volumes/certs`) for postgres data,
worker config, secrets, and Temporal mTLS certs — none tracked in git.
Running compose against a fresh `_work` checkout would silently start
postgres with an empty data dir. The data-worker k8s deploy has no
such issue.

Host bootstrap (one-time, per host):

Orchestrator:
1. Register a runner with `--labels fishsense-prod`.
2. `git clone` the repo to a persistent path (e.g. `/srv/fishsense`).
3. Set repo variable `DEPLOY_DIR` to that path under Settings ->
   Secrets and variables -> Actions -> Variables.
4. Restore `pg_volumes/`, `worker_volumes/`, `temporal_volumes/`,
   and `.secrets/` (untracked siblings of the compose files) from
   existing prod state. Populate `web_volumes/.env` (untracked) per
   the canonical shape in `deploy/web_volumes/.env.example` —
   fishsense-lite-web reads it via `env_file:` and throws on first request
   if any of the nine required keys are missing — five for the public
   landing page (FISHSENSE_API_*, LABEL_STUDIO_*) plus four for the
   Authentik OIDC gate on `/portal` (AUTH_SECRET, AUTH_AUTHENTIK_ID,
   AUTH_AUTHENTIK_SECRET, AUTH_AUTHENTIK_ISSUER). Landing stays up when
   AUTH_* are missing because the gate lives in app/portal/page.tsx,
   not on the landing route; `/portal` itself 500s. A tenth key,
   `AUTH_URL=https://fishsense.e4e.ucsd.edu`, is technically optional
   but strongly recommended — without it next-auth derives URLs from
   request headers, and the OAuth post-sign-in redirect lands at
   `http://0.0.0.0:3000/...` (the container's internal listen address)
   instead of the public hostname.

Data-worker (NRP/Kubernetes): no runner, no deploy dir. Bootstrap is
NRP-side — namespace + permanent-service exception, the three Secrets
(`fishsense-data-worker-secrets`, `fishsense-data-worker-temporal-certs`,
`ghcr-pull`), the kubeconfig (used by both the `NRP_KUBECONFIG` CI
secret and the api-worker's `[kubernetes]` config), plus the
orchestrator-side authentik prerequisite. Full list:
`deploy/k8s/data-worker/README.md`.

Three reasons for the build → release → promote → deploy split (applies
to the compose-pin PR and the kustomize-`newTag` PR alike):
1. **Race-proof promotion.** The release tag points at a specific
   commit SHA. Promote retags the image built from that exact SHA,
   not whatever happens to be `:latest`. If a newer non-release
   commit lands on main between the release-please merge and promote
   running, the wrong image can't get tagged with the release version.
2. **Don't pay the build cost twice.** build.yml already built the
   image when the release commit landed; promote.yml is a manifest
   retag (~seconds).
3. **Intentional deploy.** deploy.yml only fires when a human merges
   the auto-deploy PR. The pin diff (compose `image:` or kustomize
   `newTag:`) is reviewable in the PR before any prod change happens.

`fishsense-data-processing-workflow-worker` runs on Kubernetes
(`deploy/k8s/data-worker/`), not as a compose service — **NRP/Nautilus**
is the current target; the **Junkyard** and **Qualcomm** clusters are
longer-term targets (not ready yet; the manifests are cluster-generic
apart from the per-cluster bootstrap). Its auto-deploy PR bumps the
kustomize `newTag:` instead of a compose `image:` pin,
and the `deploy-data-worker` job `kubectl apply`s it from a
GitHub-hosted runner (see the deploy.yml routing above). **The
Deployment manifest omits `replicas`** — the api-worker owns the
replica count: each preprocess/calibration/measure parent calls
`ensure_data_worker_running_activity` (after it knows there's real
work) to scale the Deployment to `kubernetes.active_replicas`, and the
hourly `ScaleDownIdleDataWorkerWorkflow` (api-worker schedule, slot
:55) scales it back to 0 once `fishsense_data_processing_queue` has had
no running or recently-closed workflow for `kubernetes.idle_cooldown_minutes`.
Scaling no-ops when the api-worker's `[kubernetes].kubeconfig_path`
isn't set (the data-worker is then assumed always-on — local
devcontainer, pre-NRP). `active_replicas` is clamped to `[1, 4]`; >1
is a deliberate operator choice (giant single dive, or active-window
resilience on a preemption-prone cluster), never automatic. See
`deploy/k8s/data-worker/README.md`.

NRP GCs Deployments older than 2 weeks unless the namespace is on its
exceptions list — the bootstrap requests a permanent-service
exception. The worker's pods are ReplicaSet-owned so the 6-hour
bare-pod rule never applies; ConfigMaps/Secrets aren't time-GC'd.

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
`fishsense-data-processing-workflow-worker`, `fishsense-lite-web-services` —
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
DiveSlate}. Populate self-bootstraps: it calls the matching Create
activity inside the workflow body to materialize the per-dive
project (titled `"{dive.name} - <Stage> Labeling"`), then runs the
populate activity against that one project. The four
`<STAGE>_LABELING_CONFIG_XML` constants are real pasted-from-prod
XML (species XML refreshed 2026-05-05 — laser keypoints removed,
"Slate upside down" removed, Slate sub-choices expanded with
H-Slate / Tic-Tac-Toe 1..6 / V-Slate 1..4, Fish Model branch added),
so Create-on-fresh-deploy stands up a usable project immediately.
There is no discovery query or fan-out — each dive owns one project
per stage.

The populate workflows are dispatched automatically by the
preprocess parents (see "Cross-worker orchestration pattern"); manual
`temporal workflow start` is only needed for backfill of dives the
auto-chain has already cleared, or to recover from a populate that
previously errored out. Use a non-colliding workflow id for manual
runs (e.g. `populate-laser-393-manual`) so the auto-chain's
deterministic id (`populate-laser-393`) stays available for future
hourly firings.

The legacy single-project-per-stage layout (laser=73, species=70,
headtail=71, dive_slate=66, headtail-canonical=76) is grandfathered
in: dives whose images already carry label rows pointing at those
projects are excluded from the cohort selectors ("at least one image
without ANY label row"), so they aren't re-populated against the
per-dive title. New dives flow only to per-dive projects. Operators
can clean up the old shared projects manually once their incomplete
labels have all been completed.

### Laser-label validation

`SyncLabelStudioLaserLabelsWorkflow` (api-worker) dispatches a
`ValidateLaserLabelsForDiveWorkflow` child on the data-worker for
each dive whose laser labeling is fully complete (every non-superseded
`LaserLabel` has `completed=True`). The child fits a per-dive RANSAC
line through the positives, flags labels >3σ off it (with a 1-px MAD
floor for tight small-N dives), and **supersedes each flagged label**
by writing `superseded=True` back through `put_laser_label`.

Iterative-cleanup property: `get_laser_labels` filters
`superseded=False` server-side, so a re-run on the same dive sees a
smaller population, refits the line, and may flag additional
borderline labels that are now visible as outliers relative to the
cleaned inlier set. The hourly schedule re-runs against complete
dives, so this naturally tightens over time. Each per-dive run is
idempotent at the dive level — a partial failure mid-supersede leaves
the previously-superseded labels in place, and the next run picks up
where it left off.

Stage 13 + 14 consequences: laser calibration and fish measurement
both consume `LaserLabel` rows via the `superseded=False` filter, so
once a label is superseded it disappears from those pipelines too.
That's the intended behavior — bad labels should not feed
calibration. If a calibration was already computed using a
later-superseded label, the calibration row stays as-is until
something explicitly recomputes it; there's no automatic invalidation.

Open follow-up:

1. **Replace the vendored copy of `line_fit.py` with a real
   dependency.** The kernel was duplicated from
   `UCSD-E4E/2026-05-02_laser_detector` @ commit 3d5d2e8 into
   `services/fishsense-data-processing-workflow-worker/src/.../laser_label_validation/line_fit.py`
   because that repo says "early — nothing trained yet" and isn't
   published. Once the laser-detector cuts a versioned release, drop
   the vendored module and add it as a workspace / git dep in the
   data-processing worker's `pyproject.toml`. The vendored file has a
   header comment pointing at the source.

Tuning knobs if the writeback turns out too aggressive (false-positive
rate too high, watch the OUTLIER log lines):

- Raise `DEFAULT_OUTLIER_SIGMA` (currently 3.0) — straightforward
  threshold loosening.
- Raise `LABEL_NOISE_MAD_FLOOR_PX` (currently 1.0) — protects
  small-N dives where MAD collapses sub-pixel.
- Raise `LINE_CONFIDENCE_THRESHOLD` (currently 5.0) — refuses to
  supersede on dives whose line geometry isn't well-determined.
  `flag_outliers` already returns all-False for non-confident fits,
  so the practical effect is "skip more dives entirely."

Reviving a superseded label (e.g., after a labeler re-opens the LS
task and corrects the position) requires an explicit operator action
— `get_laser_label_by_label_studio_id` filters out superseded rows so
the existing sync path won't propagate the correction back to the DB.
That's the same dead-letter semantic `superseded` has had for every
other label kind; no new tooling here.

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
| 14  | measure_fish | api-worker (parent) + data-worker (child) | ported (hourly, +40min offset; idempotent as of 2026-07-17) |

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

**Species populate is decoupled (scheduled parent).** As of the
`PopulateSpeciesLabelStudioProjectParentWorkflow`, the stage-2
preprocess parent no longer chains into
`PopulateSpeciesLabelStudioProjectWorkflow` — it only writes JPEGs.
A dedicated hourly parent (schedule `populate-species-labels-workflow-schedule`,
+20 min, SKIP overlap) selects the **superseded-aware** cohort via
`GET /api/v1/dives/needing-species-population/` (HIGH + laser-valid
image with no *non-superseded* real-project species row — so dives
whose old-project rows were superseded, e.g. post hosted-LS migration,
re-enter) and fans out the populate child per dive. The populate
activity is now **idempotent** (skips images already having a
non-superseded row for the target project; supersede pass only
dead-letters *other*-project rows) and **JPEG-gated** (per-image
`ObjectStoreClient.has_processed_jpeg` — never seeds a species row
before the stage-2 JPEG exists, which would strand the image outside
the preprocess cohort). Laser/headtail/dive-slate populate are still
preprocess-dispatched (below); only species is decoupled so far.

The other three populate workflows are still dispatched automatically
as child workflows from the matching preprocess parent (stages 0.1 →
laser, 5.1 → headtail, 9 → dive-slate). After
`cleanup_raw_bytes_for_dive_activity` (Garage raw-scratch eviction),
the parent runs
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
  3. `stage_raw_bytes_for_dive_activity` — NAS → Garage `raw/` scratch.
     Stage 9 also runs `stage_slate_pdf_activity` for the slate PDF.
     NAS access is **read-only** (download only).
  4. `start_child_workflow` against the data-worker task queue. The
     child writes processed JPEGs directly to Garage.
  5. `cleanup_raw_bytes_for_dive_activity` deletes the dive's staged
     raw `.ORF` *scratch* objects from Garage (never the NAS source).
     There is no NAS archive step — the JPEGs are durable in Garage.
  6. `execute_child_workflow("Populate<Stage>LabelStudioProjectWorkflow",
     dive_id, id="populate-<stage>-{dive_id}",
     id_reuse_policy=ALLOW_DUPLICATE_FAILED_ONLY)` — on-demand
     populate child runs against the same task queue. The reuse
     policy + deterministic id deduplicate against re-firings of the
     parent on the same cohort dive — once labels start completing,
     the dive drops out of the cohort, but if the parent fires twice
     on the same dive_id (for whatever reason) the second populate
     hits `WorkflowAlreadyStartedError` and the parent catches it so
     the post-cleanup run still completes successfully.
* **Child** on data-worker (`fishsense_data_processing_queue`). Thin
  pre-input workflow that fans out per-image activities. No SDK
  calls and no NAS calls; all bytes already in Garage, all decisions
  baked into the input DTO.

**Storage = hosted Garage (S3-compatible) object store** (migrated off
the nginx file-exchange + the `fishsense_process_work` NAS share). One
bucket; the api-worker stages raw/slate scratch in and reads nothing
back; the data-worker reads scratch + writes JPEGs; Label Studio reads
the JPEGs via per-project presigned URLs. S3 access keys authenticate
from any IP — there's no IP allowlist / forward-auth, which is what
lets the data-worker run off-prem (NRP) without a stable egress IP.

NAS access lives only on the api-worker side and is **read-only** (raw
`.ORF` + slate PDF download; nothing deletes from or writes to the
NAS). The data-worker is Garage-only — narrows its blast radius and
keeps NAS credentials off the cluster.

JPEGs are the durable artifact and live in Garage (LS task data holds
`s3://bucket/<prefix>/<checksum>.JPG` URIs that LS presigns). Staged raw
`.ORF` scratch is deleted post-process because it's reproducible from
NAS. JPEG retention is a separate operational decision — see the
project memory entry.

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
* Child workflow id is deterministic (`preprocess-laser-{dive_id}`).
  The **preprocess** child is dispatched with
  `id_reuse_policy=ALLOW_DUPLICATE` (changed from
  `ALLOW_DUPLICATE_FAILED_ONLY` on 2026-07-23). FAILED_ONLY meant a
  *completed* child id could never re-dispatch, so a dive could never
  be reprocessed to pick up images that became processable after its
  first successful run — a laser validated after one-shot stage-1
  clustering, or an orphan later assigned a cluster. Those images'
  JPEGs were never produced and populate deferred them forever (prod
  dives 59/439). ALLOW_DUPLICATE is safe here: the resolver returns
  only images that still need work (finished images aren't redone) and
  the per-image activities are idempotent (S3 overwrite). A
  `WorkflowAlreadyStartedError` is now only raised when a prior child
  with the same id is still *running* (manual run overlapping the
  schedule); the parent catches it and continues to cleanup.
  The **populate** child (laser/headtail/slate parents) keeps
  `ALLOW_DUPLICATE_FAILED_ONLY` — duplicate LS `import_tasks` is the
  task-ballooning bug, so its dedup must stay. **Note:** temporalio's
  default child `id_reuse_policy` is `ALLOW_DUPLICATE`, so the populate
  child's FAILED_ONLY is set explicitly.
* Per-image activities are idempotent: S3 PutObject overwrites,
  SDK upserts.

Applied to stages 0.1, 1, 2, 5.1, 9, 13, 14 — each parent runs hourly.
Schedule slots: 0.1 at +0, 1 at +5, 2 at +15, species-populate at +20
(the decoupled `PopulateSpeciesLabelStudioProjectParentWorkflow`, just
after the +15 species-preprocess writes JPEGs), 5.1 at +30, 14 at +40,
9 at +45, 13 at +50 min — staggered so their selectors don't all hit
`dives.get()` at the top of the hour. The scale-to-zero sweeper takes
+55. `test_schedule_registration.py` pins the stagger (the four
label-studio sync schedules deliberately share +0 — they select no
dives). Per-stage cohort:

| Stage | Parent cohort definition |
|---|---|
| 0.1 | HIGH-priority + at least one image without ANY `LaserLabel` row (in any project) |
| 1   | HIGH-priority + at least one image with a *valid* `LaserLabel` (`completed=True`, `superseded=False`, `x`/`y` both set) + zero PREDICTION `DiveFrameCluster` rows |
| 2   | HIGH-priority + has PREDICTION clusters + at least one image with a *valid* `LaserLabel` whose image carries no non-sentinel `SpeciesLabel` row |
| 5.1 | HIGH-priority + at least one image with a *valid* `LaserLabel` whose image carries no non-sentinel `HeadTailLabel` row |
| 9   | HIGH-priority + `dive_slate_id` set + at least one `SpeciesLabel.content_of_image='Slate, Laser on slate'` whose image carries no `DiveSlateLabel` row at all |
| 13  | HIGH-priority + `dive_slate_id` set + no `LaserExtrinsics` + ≥2 completed `DiveSlateLabel` rows (matches the data-worker activity's `MIN_LASER_POINTS=2` precondition) |
| 14  | HIGH-priority + has `LaserExtrinsics` (own **or borrowed** via `Dive.calibration_dive_id`) + at least one *measurable* image with no `Measurement` (same predicate as the view's `measured`; keep the two in step) |

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
| `calibrated` | dive has a `LaserExtrinsics` row **of its own, or borrows one** via `Dive.calibration_dive_id` (see "Borrowed laser calibration" below) |
| `measured` | ≥1 `Measurement` for the dive AND zero *measurable* images without one. "Measurable" = a `top_three_photos_of_group` `SpeciesLabel` whose image has a valid laser label, a valid head/tail label, and a LABEL_STUDIO cluster — i.e. what `measure_fish_activity` actually attempts. Rescoped 2026-07-17; see the stage-14 notes. |

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
`images.post_cluster`. No NAS staging or object-store traffic —
clustering is pure math on `(image_id, taken_datetime)` pairs. The
cohort selector excludes dives that already have *any* PREDICTION
cluster, so this is one-shot per dive; an operator must drop partial
PREDICTION rows manually if a parent run failed mid-persist (the
cohort would otherwise skip the dive forever).

Stages 13 and 14 are structurally lighter than the preprocess parents:
pure SDK math, no NAS staging, no object-store JPEGs, no per-image
fan-out. Their selector + child-dispatch parents have only two
activity calls (selector → `start_child_workflow`); the data-worker
keeps SDK fetches inline because the math kernels need opencv +
fishsense-core, so splitting fetch/math across workers would add 5+
activity handoffs per dive for no gain.

**Borrowed laser calibration (`Dive.calibration_dive_id`).** Laser
calibration (`LaserExtrinsics`) is physically a property of the
camera+laser rig, not the dive — one slate calibration holds for every
dive shot with the same rig until it's disturbed. But stage 13 computes
`LaserExtrinsics` *per dive* from that dive's own slate labels, so a
fish-only dive with no slate frames (the slate was shot in a separate
calibration dive) can never self-calibrate and stage 14 can never
measure it. The self-referential FK `Dive.calibration_dive_id` lets such
a dive **borrow** a sibling slate dive's calibration. Resolution order,
own-wins-then-link, is centralized in `get_laser_extrinsics_for_dive`
(the endpoint `measure_fish_activity` reads via
`fs.dives.get_laser_extrinsics(dive_id)`), so the data-worker needs no
change — it transparently gets the borrowed row. Three predicates mirror
each other and must stay in step: the endpoint's fallback, the stage-14
`select_next_for_measure_fish` cohort (`has_laser_extrinsics` = own
`OR` linked), and the view's `calibrated` flag. Stage 13 is unchanged:
a linked fish dive has no `dive_slate_id`/slate labels so it's naturally
excluded from self-calibration, and a dive with its own slate always
self-calibrates (own wins). Set/clear the link via
`PUT|DELETE /api/v1/dives/{id}/calibration-source/{source_id}` (SDK:
`dives.set_calibration_source` / `clear_calibration_source`). The
management UI is the authenticated `/portal` dashboard (a follow-up to
the backend). NULL link = self-calibrate (the default for every
slate-bearing dive).

**Stage 14 became idempotent and got a schedule on 2026-07-17** (hourly,
+40 min). It used to be non-idempotent — `post_measurement` was a plain
POST and the SDK had no per-image measurement query, so a re-run on a
partially-failed dive duplicated measurements on already-bound
clusters. Both halves are fixed:

* `POST /api/v1/fish/{fish_id}/measurements` upserts on the natural key
  `(image_id, fish_id)`, backed by a `uq_measurement_image_fish`
  constraint. The key is the pair, not `image_id` alone — one frame can
  hold two fish.
* `measure_fish_activity` fetches `fish.get_measurements(dive_id)` once
  per dive and skips already-measured images *before* `_ensure_fish`,
  so a re-run causes no species/fish churn. Surfaced as
  `MeasureFishResult.skipped_already_measured`.

Scheduling was blocked on *both* halves, not just idempotency: the old
cohort predicate could never go false, so an hourly stage 14 would have
re-selected the same dives forever. With the `measured` rescope (below)
the cohort drains, so the schedule is safe. Validated on 2026-07-17
against dive 466 — 23 already-measured images skipped, 1 measured, 0
duplicates, and the dive dropped out of the cohort.

+40 rather than after calibration (+50) despite depending on it: that
would leave <5 min before the +55 scale-down sweeper. A dive calibrated
at :50 is measured at :40 the next hour, which is irrelevant at this
cadence. Each run still drains exactly one dive, so a backlog clears one
dive per hour.

Still runnable on demand for backfill — use a non-colliding workflow id
so the schedule's own id stays free. `MeasureFishParentWorkflow`:

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

1. `download_raw(checksum)` from Garage (`raw/{checksum}.ORF`).
2. (stage 9 only) `download_slate_pdf(slate_id)`
   (`slate_pdf/{slate_id}.pdf`).
3. Off-loop CPU work via `asyncio.to_thread`: rectify
   (`RectifiedImage(RawImage(bytes), intrinsics)` — rawpy + auto-gamma +
   CLAHE + `cv2.undistort`) → stage-specific overlay → `cv2.imencode`.
4. `upload_processed_jpeg(folder, checksum, jpeg_bytes)` to Garage
   (`{folder}/{checksum}.JPG`).

The client is `ObjectStoreClient` (per-worker `object_store.py`, boto3
behind `asyncio.to_thread`, path-style addressing for Garage). The
per-image JPEG prefixes are the **physical** Garage keys — the same
ones the populate activities embed in `s3://` task URIs, so there's no
virtual→physical rewrite layer anymore (this is why issue #113 is gone):

| Stage | Prefix |
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

## Garage object-store key contract

One bucket (`object_store.bucket`), content-type prefixes — the
cross-worker key contract (the analog of the old file-exchange URL
contract). Defined by the key helpers in each worker's
`object_store.py`:

```
raw/{checksum}.ORF            # api-worker stages (PUT), data-worker reads (GET); scratch
slate_pdf/{slate_id}.pdf      # api-worker stages (PUT), data-worker reads (GET); scratch (stage 9)
{jpeg_prefix}/{checksum}.JPG  # data-worker writes (PUT); durable; LS reads via presign
```

`{jpeg_prefix}` ∈ {`preprocess_jpeg`, `preprocess_groups_jpeg`,
`preprocess_headtail_jpeg`, `preprocess_slate_images_jpeg`}. Adding a
new convention is an `object_store.py` change only.

**Auth + addressing.** Garage uses S3 access keys (work from any IP)
and **path-style** addressing (no virtual-host bucket DNS) — both set
in `object_store.build_s3_client`. The api-worker key needs
rw+delete on `raw/`+`slate_pdf/` and write on the JPEG prefixes; the
data-worker key needs read on scratch + write on JPEGs; Label Studio
gets a read-only key (optional `object_store.presign_*`) to presign
the JPEGs.

**Label Studio serving.** Populate activities emit
`s3://{bucket}/{prefix}/{checksum}.JPG` into each task's `data.image`/
`data.img`. Each per-dive project gets an S3 *source* storage
registered (idempotently, `presign=True`) by
`create_or_get_label_studio_project` → `ensure_label_studio_s3_storage`,
so LS resolves those URIs to presigned GET URLs at serve time. Garage
must send CORS headers for the labeler origin so the browser can fetch
the presigned URLs directly.

**NAS safety.** The api-worker's NAS access is read-only;
`cleanup_raw_bytes_for_dive_activity` only deletes the Garage `raw/`
scratch, never the NAS source (there's a test tripwire asserting the
cleanup module imports no NAS client).

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

The `*.url` validators (incl. `object_store.endpoint_url`) use a custom
`_url_condition` (http/https + non-empty hostname) instead of
`validators.url`, because the strict library condition rejects every
Docker-internal hostname (`fishsense-api`, `temporal`, `garage` —
underscores or no TLD). Don't switch back to `validators.url`.

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
                                      image pin — deploy/incus/compose.yml
                                      for in-slot services, or the newTag
                                      in deploy/k8s/data-worker/
                                      kustomization.yaml for the data-worker)
auto-deploy PR merge → deploy.yml   (in-slot: systemctl start
                                      fishsense-selfupdate on the Incus
                                      slot's [self-hosted, fishsense]
                                      runner; data-worker: kubectl apply -k
                                      deploy/k8s/data-worker from a
                                      GitHub-hosted runner, NRP)
```

**Why the pin-bump PR merge is the deploy trigger, not the
release-please merge.** `fishsense-selfupdate` runs `nixos-rebuild
switch --flake github:UCSD-E4E/fishsense-lite#fishsense --refresh`, so
the slot converges to whatever is *on main* — including
`deploy/incus/compose.yml`'s `image:` pins. When the release-please PR
merges, main still carries the previous pins and the `:v<new>` images
don't exist yet (promote.yml retags them minutes later, on
`release: published`). A converge fired then would redeploy the old
images. promote.yml's pin-bump PR is the first point at which main
describes the release, so its merge is the trigger.

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
`deploy/incus/compose.yml`; for `fishsense-data-processing-workflow-worker`
(which runs on NRP/Kubernetes, not in the slot) it's the `newTag:` in
`deploy/k8s/data-worker/kustomization.yaml`. Branch name pattern:
`auto-deploy/<package>-<version>` either way (deploy.yml routes by
prefix).

The legacy `deploy/compose*.yml` pins are **not** bumped — nothing
deploys that host anymore. Adding a new in-slot service means giving it
a stanza in `deploy/incus/compose.yml` *and* adding it to promote.yml's
`deployable` case list, or its release will silently never roll out
(promote logs a `::notice::` and opens no PR).

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
unrelated edits don't trigger it). Plus a `workflow_dispatch` (`target`:
`incus` | `data-worker`) for manual re-deploys. Two jobs route by branch
name:

- `auto-deploy/fishsense-data-processing-workflow-worker-*` ->
  `deploy-data-worker` on `ubuntu-latest` (GitHub-hosted; kubectl is
  preinstalled): writes the `NRP_KUBECONFIG` repo secret to a temp
  kubeconfig and runs `kubectl apply -k deploy/k8s/data-worker` +
  `kubectl rollout status`. No persistent ops dir, no docker — the
  kustomization is self-contained and config/certs live in cluster
  ConfigMaps/Secrets.
- any other `auto-deploy/*` -> `deploy-incus` on
  `[self-hosted, fishsense]` — the slot's own runner, auto-provisioned
  and auto-registered by the platform because the tenant flake sets
  `repo = "UCSD-E4E/fishsense-lite"` (ADR 0022). Its one step is
  `systemctl start --no-block fishsense-selfupdate` (polkit-authorized
  for the non-admin `gh-runner` user). The unit runs `nixos-rebuild
  switch --flake github:UCSD-E4E/fishsense-lite#fishsense --refresh`, so
  the slot pulls the flake from GitHub itself — nothing is checked out
  onto it. The `deploy-incus` job carries a `concurrency: deploy-incus`
  group (`cancel-in-progress: false`) so two pin-bump PRs merged back to
  back converge in sequence rather than the second cancelling the first.

Never give a job a bare `runs-on: self-hosted` — the `fishsense` label
is what keeps GitHub-hosted work (the NRP deploy) off the tenant slot.

**Two other convergence paths beyond the pin-bump PR merge.** (1) A
manual `deploy.yml` `workflow_dispatch` (`target: incus`) — for
config-only `deploy/incus/` changes, which cut no release and so open no
`auto-deploy/*` PR. (2) The **nightly `system.autoUpgrade`** (04:00,
krg-infra #460) rebuilds the slot from `github:UCSD-E4E/fishsense-lite#fishsense`,
so anything on main rolls out within a day unattended (`allowReboot=false`).

**Committed config applies only because the converge force-recreates.**
The composeStack unit runs `up -d --force-recreate` (krg-infra
`recreateOnConfigChange`, their #459 / our krg-infra#458). A bind-mounted
config file is a project-dir store symlink whose target changes on edit,
but plain `up -d` sees no compose-spec change and won't re-read it — a
stale bind mount once crash-looped the api-worker on old config while the
fix "deployed" with no effect. Force-recreate fixes that, at the cost of
recreating the whole stack (postgres included) on any changed converge.

**The converge is a trigger, not a clean CI gate.** `fishsense-selfupdate`
restarts the very runner executing the job (its first action is
`systemctl stop github-runner-fishsense`). The step therefore uses
`systemctl start --no-block`: it returns 0 the instant the converge is
*enqueued*, so the job's green means "converge fired", never "containers
are healthy". Without `--no-block` a blocking `systemctl start` gets
SIGINT'd (exit 130, false red) whenever `nixos-rebuild` outlives the
runner's graceful-stop window — observed live on PR #238 (red at ~6min)
vs #240 (green at 34s), same code path, non-deterministic.

**`fishsense-selfupdate`'s exit code does NOT mean the deploy failed.**
Do not "verify the converge" with `systemctl status fishsense-selfupdate`
— an earlier revision of this file said to, and it is wrong.
`switch-to-configuration`'s `get_active_units` keeps every unit whose
`state != "inactive"` and exits **4** if any of them is `failed` — even a
unit the switch never touched, and even though it has *already applied
the whole configuration*. Measured during the 2026-07-17 outage:

```
api-worker container started:   04:09:51Z   <- new config live
switch-to-configuration exit 4: 04:10:32Z   <- 41s later, "failed"
```

The pin at the built commit was the version that ended up running, so
that converge deployed correctly and then reported failure, purely
because `github-runner-fishsense` was in a failed state. One lingering
failed unit therefore poisons every subsequent switch on the slot
(krg-infra#496 traced this through `switch-to-configuration-ng`; a real
fix needs the unit to never *end* a switch failed, and is still open).

Consequences, both counter-intuitive:

* A **red** converge does not mean nothing deployed. Check what is
  actually running (`docker ps` image tags vs the pin on main), not the
  exit code.
* Gating CI on `systemctl show fishsense-selfupdate -p Result` is
  therefore NOT the fix for the false green — it would false-red every
  working deploy for as long as any unit is failed (which is the state
  the slot is in whenever the runner can't register).

**`verify-incus` is the deploy gate** (#297). It used to poll the public
endpoints, which answer from the OLD containers when a converge fails —
so it went green on a deploy that never happened, alongside a green
`deploy-incus`. It now asserts every `ghcr.io/ucsd-e4e/*` pin the slot
deployed matches the pin on main, and fails the run otherwise. All must
match; a partial match is a failed deploy.

The signal is the compose spec the stack unit actually deployed:
`fishsense.service`'s `ExecStart` names its `/nix/store` path (world-
readable), and the unit only reaches `Result=success` if
`up -d --force-recreate` brought that spec up — both are checked, since
a spec can land on disk while the containers crash-loop. Not `docker ps`:
`/var/run/docker.sock` is `0660 root:docker` and `gh-runner` isn't in
that group (adding it is root-equivalent, and the runner is platform-
provisioned anyway).

It runs on the slot's runner deliberately. The converge stops that runner
gracefully — after `deploy-incus`'s job ends — and restarts it on the way
out, so the gate can't be dispatched mid-converge; and a converge that
dies leaving the runner down never lets it start, so the run times out.
That is the correct signal: nothing can deploy until the runner returns.
This reverses the earlier "report-only, false-reds not acceptable"
stance — a false red is loud and gets investigated, the false green
quietly asserted a deploy that hadn't happened.

**If the runner is down, deploys queue — converge by hand.** The
`deploy-incus` job needs the slot's own runner, so a downed runner
blocks the pipeline; but a converge does not need the job. Run what the
job would have run:

```
ssh krg-admin@krg-nat.ucsd.edu \
  'incus exec fishsense --project fishsense -- \
     systemctl start --no-block fishsense-selfupdate'
```

It pulls current main and deploys normally. Verified 2026-07-17: with
the runner failed, this shipped `api-workflow-worker` v1.39.0 → v1.40.0
and registered the stage-14 schedule, while `fishsense-selfupdate`
reported `Result=exit-code ExecMainStatus=4`. Confirm with `docker ps`
image tags, not the exit code. The nightly `autoUpgrade` does the same
thing unattended, so waiting a day also works.

**Runner registration is the usual reason a converge exits 4** (and the
usual reason deploys then queue forever: the failed switch leaves the
runner down, and the runner is what picks up the next deploy job).
krg-infra minted a fresh registration token per fleet deploy and pushed
it to the tenant; the upstream module diffs the token file's *contents*,
so a fresh token guaranteed "Config has changed, removing old runner
state" → re-register → any GitHub hiccup fails activation. Registration
tokens live ~1h, so converges shortly after a krg deploy registered fine
and converges long after got `404 Not Found` — which reads as a random
flake but is a clock. Fixed fleet-side by krg-infra#496 (push a token
only when `.credentials` is absent). If the runner is down: it is
platform-provisioned (ADR 0022), so don't hand-register it — a krg fleet
deploy re-mints for any instance missing `.credentials`.

**A pushed token is inert until the runner restarts to consume it**, and
that is the half that is *ours*. krg stages the token to
`/var/lib/krg/github-runner/registration-token`; the runner reads it on
start. Nothing reads it on its own — upstream's module hardcodes
`Restart = "no"`, so a runner that 404'd once stays dead while perfectly
good tokens pile up underneath it. krg-infra#496 adds
`Restart = lib.mkForce "on-failure"` + `RestartSec = 30`, but that lives
in `nix/modules/tenant.nix` — Axis B, so it only reaches the slot when
`flake.lock`'s krg-infra pin moves (`nix flake update krg-infra`;
`update-flake.yml` does it Mondays 08:00 UTC). On an old pin, a token
delivery fixes nothing.

To recover a dead runner: converge (see the hand-converge box above)
*within the token's ~1h life*. Verified 2026-07-17 — token staged 06:12,
converge at 06:53, `Listening for Jobs` at 06:54, and the converge itself
went back to `Result=success` because the failed unit was the only thing
producing exit 4. Waiting for the 11:00 nightly would have found the
token expired and 404'd again.

**Check the right file.** `$STATE_DIRECTORY/.new-token`
(`/var/lib/github-runner/fishsense/.new-token`) is the *module's* copy,
written by its own `copy_tokens()` when the runner **starts** — krg never
writes it. Its mtime tracks the last runner start, not the last push, so
on a dead runner it is frozen and cannot tell you whether a token was
delivered. Ask `/var/lib/krg/github-runner/registration-token` instead.
(Learned the hard way: an entire "their fix isn't pushing to us"
diagnosis was built on the wrong file's mtime, plus a fleet-deploy run's
*start* time misread as its finish — the push lands in a later job,
~10min in.)

Until the slot is bootstrapped the converge job fails; the
`deploy-data-worker` job runs but fails fast until `NRP_KUBECONFIG` is
set. The retired `deploy-orchestrator` job (docker compose on a
`fishsense-prod` runner that was never registered) was folded away when
`auto-deploy.yml` merged into `deploy.yml`; `deploy/compose*.yml` was
deleted with it, and the repo variable `DEPLOY_DIR` is now unused —
delete it from Settings -> Secrets and variables -> Actions.

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

`deploy/incus/compose.yml` is the home for the `fishsense-*` worker
services that run in the slot: `fishsense-api-workflow-worker` and
`fishsense-backup-worker`. (Workers consume Temporal — now krg-prod's,
over mTLS — but aren't part of the cluster.) The backup worker reads
its non-secret config from the committed
`deploy/incus/worker_volumes/backup_worker/config/settings.toml`; its
postgres + NAS credentials come from the vault-agent render
(`/run/tenant/secrets/backup-postgres.env`, see
`deploy/incus/secrets.nix`), so OpenBao must be seeded before the
service will start successfully.

The pre-Incus orchestrator host's compose files (`deploy/compose.yml` +
`compose.orchestrator/temporal/workers/superset.yml`) and their
bind-mount `*_volumes/` dirs were deleted when that host was
decommissioned. `git log --diff-filter=D -- deploy/compose.yml` recovers
them.

**Don't use `git merge-base --is-ancestor` to check whether a fix
shipped.** Squash merges are enabled here, so a merged PR's *content* is
in main under a new commit while its original commits are not ancestors
of anything — the ancestry test returns a false negative. Same trap in
reverse when checking a release tag: ask whether the *content* is there
(`git show <tag>:<path> | grep`, or diff the file), not whether a commit
is reachable.

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

`api.fishsense.e4e.ucsd.edu` (renamed from `orchestrator.` at the Incus
migration) is fronted by the slot's inner Traefik with a `forwardAuth`
middleware pointing at the co-located authentik outpost (see
[deploy/incus/compose.yml](deploy/incus/compose.yml)).
`fishsense-api-sdk.Client` uses HTTP Basic auth, so a request from a
dev box gets a 302 redirect to authentik's OAuth flow and
`raise_for_status()` blows up — even with valid credentials. Workers
running in the slot hit fishsense-api on the interior docker network
and skip the proxy entirely.

For dev access, in rough order of effort:
1. `incus exec` into the slot and point `fishsense_api.url` at the
   interior address.
2. Bypass the API and read Postgres directly (host is in
   `deploy/incus/fishsense_api_volumes/config/settings.toml`).
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

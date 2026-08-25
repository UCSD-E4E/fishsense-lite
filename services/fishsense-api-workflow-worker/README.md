# fishsense-api-workflow-worker

Temporal worker for api-side orchestration — Label Studio sync
workflows, Create/Populate × {Laser, Species, HeadTail, DiveSlate}
project workflows, and the hourly parents (preprocess 0.1 / 2 / 5.1 /
9, clustering 1, calibration 13, measurement 14) that dispatch work to
the data-worker. It also owns the data-worker's replica count on NRP:
parents call `ensure_data_worker_running_activity` before dispatching
a child, and `ScaleDownIdleDataWorkerWorkflow` scales the Deployment
back to 0 when its task queue is quiet (see `[kubernetes]` config
below — a no-op when that's unset). Talks to `fishsense-api` (via
`fishsense-api-sdk`), Label Studio (via `label-studio-sdk`), the E4E
NAS, and (when configured) the NRP k8s API.

Task queue: `fishsense_api_queue`.

## Workflows

| Workflow | Cadence | Purpose |
|---|---|---|
| `SyncLabelStudio{Laser,HeadTail,Species,DiveSlate}LabelsWorkflow` | every 1 h, +0 | Pull labels from Label Studio → write to fishsense-api. All four share +0 deliberately: they select no dives, so they cannot race each other. |
| `ReconcileLabelingConfigsWorkflow` | every 1 h | Keep each per-dive project's labeling-config XML in step with the stored constant. |
| `Create<Stage>LabelStudioProjectWorkflow` × 4 | on-demand | Idempotently create **the dive's** LS project — title-lookup or create from the stored labeling-config XML. |
| `Populate<Stage>LabelStudioProjectWorkflow(dive_id)` × 4 | on-demand | Materialize the per-dive project (calls the Create activity), then import that one dive's tasks. **Projects are per-dive**, so there is no discovery query and no fan-out. Idempotent: it selects only images without a non-sentinel label row, and `import_tasks_and_record_labels` dedupes by URL against tasks already in the project. |
| `PopulateLaserLabelStudioProjectParentWorkflow` | every 1 h, +0 | Fans the laser populate child out over the cohort needing it. |
| `PopulateSpeciesLabelStudioProjectParentWorkflow` | every 1 h, +20 | Species populate is **decoupled** from the stage-2 preprocess parent; it selects the superseded-aware cohort and is JPEG-gated per image. |
| `PreprocessLaserImagesParentWorkflow` | every 1 h, +0 (`overlap=SKIP`) | Stage-0.1: select → resolve → stage raw `.ORF`s NAS→Garage → wake the data-worker → dispatch `PreprocessLaserImagesWorkflow` → delete the staged raw scratch → dispatch the populate child. |
| `ClusterDiveFramesParentWorkflow` | every 1 h, +5 (`overlap=SKIP`) | Stage-1 (laser-valid dive without PREDICTION clusters). Selector → resolver → dispatch `DiveFrameClusteringWorkflow` → persist PREDICTION clusters. No NAS or object-store traffic — clustering is pure maths on `(image_id, taken_datetime)`. |
| `PreprocessSpeciesImagesParentWorkflow` | every 1 h, +15 (`overlap=SKIP`) | Stage-2 (PREDICTION clusters + laser-valid image without a species row). Writes JPEGs only; populate is separate. |
| `PreprocessHeadtailImagesParentWorkflow` | every 1 h, +30 (`overlap=SKIP`) | Stage-5.1 (laser-valid image without a head/tail row). |
| `ComputeLaserDepthsParentWorkflow` | every 1 h, +35 (`overlap=SKIP`) | Per-image `LaserDepth`. Selects on **provenance mismatch**, not absence, so a recalibration drains as an ordinary cohort. |
| `MeasureFishParentWorkflow` | every 1 h, +40 (`overlap=SKIP`) | Stage-14. Idempotent since 2026-07-17 (`post_measurement` upserts on `(image_id, fish_id)`); drains one dive per run. |
| `PreprocessSlateImagesParentWorkflow` | every 1 h, +45 (`overlap=SKIP`) | Stage-9. Also stages the slate template PDF before dispatch. |
| `PerformLaserCalibrationParentWorkflow` | every 1 h, +50 (`overlap=SKIP`) | Stage-13: select → dispatch `PerformLaserCalibrationWorkflow`. No NAS or object-store staging. |
| `ScaleDownIdleDataWorkerWorkflow` | every 1 h, +55 (`overlap=SKIP`) | Scale the NRP data-worker Deployment to 0 when its queue is idle — **or when it is busy but *wedged*** (`spec.replicas > 0` with nothing Ready), which is how a crash-looping worker stops holding GPUs 24/7. No-op when k8s scaling isn't configured. |
| `IngestDiveWorkflow(request)` | on-demand | Ingest one NAS folder into a `Dive` + its `Image` rows. See [docs/ingest.md](../../docs/ingest.md). |
| `VerifyDiveChecksumsWorkflow` / `VerifyAllDivesChecksumsWorkflow` | on-demand | Re-hash existing rows against the NAS and report. Read-only, with a source tripwire. |
| `UpdateDiveImageGroupsWorkflow(dive_id)` | on-demand | Stage-6.1: reconcile species labels into LABEL_STUDIO clusters. Refuses to re-run when clusters already exist — the cluster API has no DELETE, so a re-POST would double-count. |
| `PredictSlateImagesParentWorkflow`, `BackfillSlatePredictionsWorkflow` | **dormant** | Model-assisted slate labeling, shut down 2026-08-03 — the ECC gate does not transfer out of distribution. Registered so a future evaluation can start it by hand; its schedule is *actively deleted* at startup. |

The stagger is not cosmetic: it keeps the parents' selectors from all hitting
`dives.get()` at the top of the hour, and it is pinned by
`test_schedule_registration.py`. There is **no NAS archive step** — JPEGs are
durable in Garage, and the api-worker's NAS access is read-only (a tripwire
asserts the cleanup module imports no NAS client).

Schedules are auto-registered at worker startup if missing, so the
first deploy creates them and subsequent deploys are no-ops. To change
a cadence or workflow type, delete the schedule via
`temporal schedule delete <id>` and let the next worker startup
recreate it (refusing to update in-place is intentional — a config typo
would silently retire the schedule otherwise; same pattern as the
backup worker).

## Activities

Per-workflow `activities/*.py` modules. Shared helpers in
`label_utils.py`, `utils.py`, and `populate_utils.py`; the NRP
data-worker scaling helpers live in `activities/k8s_scaling.py`
(`resolve_scaling_config`, `apps_v1_api`, `set_deployment_replicas`),
used by `ensure_data_worker_running_activity` and
`scale_down_data_worker_if_idle_activity`. See `CLAUDE.md` for the
full notebook-port status table.

## Required config (`E4EFS_` prefix — env vars or settings.toml)

```
E4EFS_TEMPORAL__HOST, E4EFS_TEMPORAL__PORT
E4EFS_TEMPORAL__TLS=true|false
E4EFS_TEMPORAL__CLIENT_CERT, E4EFS_TEMPORAL__CLIENT_PRIVATE_KEY  # when tls=true
E4EFS_TEMPORAL__SERVER_ROOT_CA_CERT, E4EFS_TEMPORAL__DOMAIN      # optional
E4EFS_LABEL_STUDIO__URL, E4EFS_LABEL_STUDIO__API_KEY
E4EFS_E4E_NAS__URL, E4EFS_E4E_NAS__USERNAME, E4EFS_E4E_NAS__PASSWORD
E4EFS_FISHSENSE_API__URL
E4EFS_FISHSENSE_API__USERNAME, E4EFS_FISHSENSE_API__PASSWORD     # optional
# NRP data-worker scaling — all optional; scaling is OFF (the activities
# no-op) unless kubeconfig_path is set:
E4EFS_KUBERNETES__KUBECONFIG_PATH    # path to the NRP kubeconfig (file mounted into the pod)
E4EFS_KUBERNETES__NAMESPACE          # required when kubeconfig_path is set
E4EFS_KUBERNETES__DEPLOYMENT_NAME    # default: fishsense-data-processing-workflow-worker
E4EFS_KUBERNETES__ACTIVE_REPLICAS    # default 1; clamped to [1, 4]
E4EFS_KUBERNETES__IDLE_COOLDOWN_MINUTES  # default 15
```

`general.max_workers` (default 4) caps the activity thread pool.

Dynaconf eagerly validates *every* `Validator` on first attribute
access of `settings`, not lazily per setting — tests that import any
activity module must plumb env values for all required settings even
if the test only uses one of them.

## Local development

Inside the devcontainer, `deploy/compose.local.yml` already runs
Temporal + fishsense-api + Label Studio + nginx, and the `dev`
container exports the `E4EFS_*` config pointing at them, so just run
the worker:

```
uv run --package fishsense-api-workflow-worker fishsense_api_workflow_worker
```

It registers its schedules and starts polling `fishsense_api_queue`.
(`k8s` scaling no-ops — `kubernetes.kubeconfig_path` isn't set
locally.) Outside the devcontainer, run your own
`temporal server start-dev` and set `E4EFS_TEMPORAL__HOST=localhost`
(plus the other `E4EFS_*` config — see "Required config" above).

## Tests

```
./check.sh unit           # default markers, mocks only — fast
./check.sh integration    # -m integration; needs the local stack
```

Integration tests exercise the populate / create activities against
the real Label Studio container at `http://label-studio:8080`. The
container is provisioned in `deploy/compose.local.yml` with a
hard-coded admin token (`fishsense_local_test_token_42`) so tests can
authenticate without going through the LS UI — the token is also
mirrored into the `dev` container's env (`E4EFS_LABEL_STUDIO__API_KEY`)
so newly-spawned tests pick it up automatically. The `dev` container
must be recreated (`docker compose -f deploy/compose.local.yml up -d
--force-recreate dev`) the first time you pull these env vars from
upstream — bare `docker compose up -d` doesn't pick up env changes on
already-running containers.

Each integration test creates its own LS project (UUID-suffixed
title, ≤ 50 chars per LS limit) and deletes it on teardown. State
between tests is fully isolated; a `docker compose down -v label-studio`
also resets the LS volume if you want a clean slate.

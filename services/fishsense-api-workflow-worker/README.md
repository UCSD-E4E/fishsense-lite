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
| `SyncLabelStudioLaserLabelsWorkflow` | every 1 h | Pull laser labels from Label Studio projects → write to fishsense-api. |
| `SyncLabelStudioHeadTailLabelsWorkflow` | every 1 h | Same shape for head/tail labels. |
| `Create<Stage>LabelStudioProjectWorkflow` × 4 | on-demand | Idempotently create the LS project for a stage (laser / species / headtail / dive_slate). Title-lookup or create from labeling-config XML. |
| `Populate<Stage>LabelStudioProjectWorkflow(dive_id)` × 4 | on-demand | Query SQL for active LS projects (`incomplete=True`), fan out task imports across them with `Semaphore(4)`. |
| `PreprocessLaserImagesParentWorkflow` | every 1 h (`overlap=SKIP`) | Stage-0.1 orchestrator: select → resolve → stage raw `.ORF`s NAS→file-exchange → dispatch `PreprocessLaserImagesWorkflow` → archive JPEGs file-exchange→NAS → cleanup raw `.ORF`s. |
| `ClusterDiveFramesParentWorkflow` | every 1 h, +5 min (`overlap=SKIP`) | Stage-1 orchestrator (laser-valid dive without PREDICTION clusters). Selector → resolver → dispatch `DiveFrameClusteringWorkflow` → persist PREDICTION clusters via SDK. No NAS or file-exchange staging. |
| `PreprocessSpeciesImagesParentWorkflow` | every 1 h, +15 min (`overlap=SKIP`) | Stage-2 orchestrator (PREDICTION cluster + laser-valid image without species label). Same five-step shape. |
| `PreprocessHeadtailImagesParentWorkflow` | every 1 h, +30 min (`overlap=SKIP`) | Stage-5.1 orchestrator (laser-valid image without head/tail label). Same shape. |
| `PreprocessSlateImagesParentWorkflow` | every 1 h, +45 min (`overlap=SKIP`) | Stage-9 orchestrator (slate-marked species labels lacking slate labels). Also stages the slate template PDF (`stage_slate_pdf_activity`) before dispatch. |
| `PerformLaserCalibrationParentWorkflow` | every 1 h, +50 min (`overlap=SKIP`) | Stage-13 orchestrator: select → dispatch `PerformLaserCalibrationWorkflow`. No NAS/file-exchange staging. |
| `MeasureFishParentWorkflow` | on-demand (`overlap=SKIP` when scheduled) | Stage-14 orchestrator: select → dispatch `MeasureFishWorkflow`. Drains one dive per run; not scheduled (measurement isn't idempotent — see CLAUDE.md). |
| `ScaleDownIdleDataWorkerWorkflow` | every 1 h, +55 min (`overlap=SKIP`) | Scale the NRP data-worker Deployment to 0 if `fishsense_data_processing_queue` has had no running or recently-closed workflow for `kubernetes.idle_cooldown_minutes`. No-op when k8s scaling isn't configured. |

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

For an end-to-end Temporal dev loop without the full stack:

```
temporal server start-dev      # terminal 1: local Temporal
uv run --package fishsense-api-workflow-worker fishsense_api_workflow_worker  # terminal 2: this worker
```

Inside the devcontainer the rest of the stack is already up via
`deploy/compose.local.yml`, so only the worker needs to be run
manually.

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

# fishsense-api-workflow-worker

Temporal worker for api-side orchestration — currently the
Label Studio sync workflows and the Superset dashboard config writer.
Talks to `fishsense-api` (via `fishsense-api-sdk`), Label Studio (via
`label-studio-sdk`), and the E4E NAS.

Task queue: `fishsense_api_queue`.

## Workflows

| Workflow | Cadence | Purpose |
|---|---|---|
| `SyncLabelStudioLaserLabelsWorkflow` | every 1 h | Pull laser labels from Label Studio projects → write to fishsense-api. |
| `SyncLabelStudioHeadTailLabelsWorkflow` | every 1 h | Same shape for head/tail labels. |
| `UpdateDashboardConfigWorkflow` | every 1 h | Render the Superset dashboard config from current api state. |
| `Create<Stage>LabelStudioProjectWorkflow` × 4 | on-demand | Idempotently create the LS project for a stage (laser / species / headtail / dive_slate). Title-lookup or create from labeling-config XML. |
| `Populate<Stage>LabelStudioProjectWorkflow(dive_id)` × 4 | on-demand | Query SQL for active LS projects (`incomplete=True`), fan out task imports across them with `Semaphore(4)`. |

Schedules are auto-registered at worker startup if missing, so the
first deploy creates them and subsequent deploys are no-ops. To change
a cadence or workflow type, delete the schedule via
`temporal schedule delete <id>` and let the next worker startup
recreate it (refusing to update in-place is intentional — a config typo
would silently retire the schedule otherwise; same pattern as the
backup worker).

## Activities

Per-workflow `activities/*.py` modules. Shared helpers in
`label_utils.py`, `utils.py`, and `populate_utils.py`. The
populate stages (0.3 / 4 / 5.3 / 11) are ported; the sync stages
(4.2 / 12) are partial; calibration / measurement / dive-image-groups
(6.1 / 13 / 14) are not yet ported. See `CLAUDE.md` for the full port
status table.

## Required env (`E4EFS_` prefix)

```
E4EFS_TEMPORAL__HOST, E4EFS_TEMPORAL__PORT
E4EFS_TEMPORAL__TLS=true|false
E4EFS_TEMPORAL__CLIENT_CERT, E4EFS_TEMPORAL__CLIENT_PRIVATE_KEY  # when tls=true
E4EFS_TEMPORAL__SERVER_ROOT_CA_CERT, E4EFS_TEMPORAL__DOMAIN      # optional
E4EFS_LABEL_STUDIO__URL, E4EFS_LABEL_STUDIO__API_KEY
E4EFS_E4E_NAS__URL, E4EFS_E4E_NAS__USERNAME, E4EFS_E4E_NAS__PASSWORD
E4EFS_FISHSENSE_API__URL
E4EFS_FISHSENSE_API__USERNAME, E4EFS_FISHSENSE_API__PASSWORD     # optional
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

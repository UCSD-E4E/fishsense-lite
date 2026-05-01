# deploy/

Docker-compose stack for FishSense Lite — both the local devcontainer
and the prod orchestrator host. The prod data-processing-worker host
runs a separate compose file that lives elsewhere (held off auto-deploy).

## Compose layout

| File | Purpose |
|---|---|
| `compose.yml` | Top-level prod stack. `include:`s the four siblings below + defines `postgres`, `qcomm-static-file-server`, `mafl`. Pulls `prometheus_network` and `traefik_proxy` as external networks. |
| `compose.orchestrator.yml` | `fishsense-api` + nginx `static_file_server` (file-exchange DAV). Behind Traefik + `authentik@docker` middleware. |
| `compose.temporal.yml` | Temporal cluster (history, frontend, matching, worker, UI). |
| `compose.workers.yml` | `fishsense-*` workers running on the orchestrator host: `fishsense-api-workflow-worker`, `fishsense-backup-worker`. (Workers consume Temporal but aren't part of the cluster.) |
| `compose.superset.yml` | Superset + redis + worker / beat. |
| `compose.local.yml` | Self-contained local devcontainer stack — postgres, temporal, fishsense-api (pinned image), nginx static_file_server, label-studio, dev container. **Not** layered on `compose.yml`; prod's authentik / mTLS / letsencrypt coupling is intentionally absent. |
| `compose.dev.yml` | Dev-only compose extras (used inside the devcontainer for tooling). |

`compose.local.yml` and the prod stack are intentionally separate
files. Prod is fronted by Traefik + authentik + letsencrypt + mTLS to
Temporal; trying to make a single set of compose files that boots both
on a laptop and on the orchestrator was messier than splitting them.

## Local development (devcontainer)

The repo ships a VS Code devcontainer that boots the local stack and
gives you a shell inside it.

1. `cp deploy/.env.local.example deploy/.env` and fill in:
   - `HOST_REPO_PATH` — output of `realpath ..` from this directory.
     Required so the repo path is the same inside and outside the
     container (Claude Code's project memory keys off this).
   - `DOCKER_GID` — output of `getent group docker | cut -d: -f3`.
   - `FISHSENSE_DUMP_PATH` — optional path to a `pg_dump -Fc` of prod.
     Leave commented to boot with empty DBs (fine for shape work).
2. Open the repo in VS Code → "Reopen in Container".

The first boot of `postgres` runs `pg_volumes/scripts.local/00_restore.sh`,
which `pg_restore`s the dump into a named docker volume (NOT under the
repo — the data is large). To start clean later:
`docker volume rm fishsense-local_pg_data`.

`label-studio` boots with a hard-coded admin token
(`fishsense_local_test_token_42`) so workflow integration tests can
authenticate without going through the LS UI. The `dev` container
exports the same token as `E4EFS_LABEL_STUDIO__API_KEY`.

The `dev` container must be recreated the first time you pull new env
vars from upstream:

```
docker compose -f deploy/compose.local.yml up -d --force-recreate dev
```

Bare `docker compose up -d` won't pick up env changes on already-running
containers.

## Production deploy

Prod runs `compose.yml` (which `include:`s the four sibling files) on
the orchestrator host. **Bringing it up by hand is not the path** —
the auto-deploy pipeline handles version bumps via reviewable PRs.
See [.github/workflows/deploy.yml](../.github/workflows/deploy.yml)
and the "CI pipeline" section of [CLAUDE.md](../CLAUDE.md).

### Host bootstrap (one-time, ops)

The deploy workflow does **not** check the repo out into the runner's
default `_work` directory. It operates on a persistent ops-managed
directory on the host. This matters because the compose files use
relative bind mounts (`./pg_volumes/`, `./worker_volumes/`, `./.secrets/`,
etc.) for postgres data, worker config, secrets, and Temporal env
files — none of which are tracked in git. Running compose against a
fresh `_work` checkout would silently start postgres with an empty
data dir.

1. Register a self-hosted GitHub runner with `--labels fishsense-prod`,
   co-located with the docker engine that will run the stack.
2. `git clone` this repo to a persistent path (e.g. `/srv/fishsense`).
3. Set repo variable `DEPLOY_DIR` (Settings → Secrets and variables →
   Actions → Variables) to that absolute path.
4. Restore the untracked sibling directories from existing prod state:
   - `pg_volumes/data/` — postgres data dir.
   - `pg_volumes/config/` — `postgres.conf`, etc.
   - `pg_volumes/scripts/` — init scripts.
   - `temporal_volumes/certs/` — mTLS certs (read by workers too).
   - `worker_volumes/config/` — `settings.toml` + `.secrets.toml` for
     `fishsense-api-workflow-worker`.
   - `worker_volumes/backup_config/` — same for `fishsense-backup-worker`.
   - `worker_volumes/backup_worker/logs/` — log volume.
   - `mafl_volumes/data/` — mafl config + dashboard config.
   - `superset_volumes/`, `qcomm_static_file_server_volumes/`,
     `static_file_server_volumes/`, `fishsense_api_volumes/` — see the
     respective compose files for the bind-mount paths.
   - `.secrets/postgres_admin_password.txt`
   - `.secrets/temporal_database_password.env` (sets `POSTGRES_PWD`)
   - `.secrets/temporal_ui.env` (sets `TEMPORAL_AUTH_CLIENT_ID` /
     `_SECRET`)
   - `.secrets/superset-env` (optional Superset overrides)
5. Ensure the external networks `traefik_proxy` and `prometheus_network`
   exist on the host.
6. Set `USER_ID` / `GROUP_ID` in `.env` so the postgres container runs
   as the host owner of `pg_volumes/data/`.

### Manual rollout (workers off auto-deploy)

`fishsense-data-processing-workflow-worker` runs on a different host
whose compose file isn't in this repo. `promote.yml` still publishes
`:v<version>` tags for it, so manual rollout on that host is just:

```
docker compose pull fishsense-data-processing-workflow-worker
docker compose up -d fishsense-data-processing-workflow-worker
```

If the host is upgrading from the polyrepo worker, **rename
`DYNACONF_*` env vars to `E4EFS_*`** before redeploying. Dynaconf will
silently ignore the old prefix.

### Rollout operational notes

#### Settings-file changes ride the deploy atomically

`deploy.yml` does `git pull --ff-only origin main` BEFORE
`docker compose pull && up -d`. Schema changes to
`worker_volumes/config/settings.toml` (the in-repo copy) flow to the
host atomically with the image-pin bump — no drift window where the
new image runs against the old settings or vice versa.

Caveat: `--ff-only` will refuse if ops manually edited the host's
checkout (e.g. tweaked `settings.toml` directly). Always commit
config changes through PRs and let auto-deploy carry them in.

#### Temporal schedules are idempotent-as-create

Workers register schedules at startup via `ensure_schedule` (treats
`ScheduleAlreadyRunningError` as success). First deploy creates the
schedule; subsequent deploys leave existing schedules alone.

To change a schedule's cron / timeout / args, manually delete then
let the next worker restart recreate it:

```
temporal schedule delete <schedule-id>
```

A code-side timeout bump WILL NOT take effect on its own.

#### First-run sync behavior

Hourly `SyncLabelStudio*LabelsWorkflow` activities use a per-(kind,
project) cursor (`LabelStudioSyncCursor`). On the first deploy of a
new sync the cursor is NULL → the activity processes every existing
label. Subsequent runs are incremental. Rough backlog sizes as of
2026-05-01:

| Kind | Projects | Backlog | First-run estimate |
|---|---|---|---|
| `laser` | varies | varies | already deployed |
| `headtail` | varies | varies | already deployed |
| `dive_slate` | ~1 | tens of tasks | seconds |
| `species` | 8 | ~1218 tasks | ~5 min |

After a new sync first deploys, watch `fishsense_api_queue` in
Temporal UI. If a per-project activity trips its 2h
`schedule_to_close_timeout`, raise the timeout in the workflow file
and cut a release.

#### Stage 6.1 (UpdateDiveImageGroupsWorkflow) — on-demand, ordering matters

Stage 6.1 reconciles species labels into LABEL_STUDIO frame clusters
that stage 14 measurement reads. It is on-demand (no schedule),
triggered per dive:

```
temporal workflow start \
  --type UpdateDiveImageGroupsWorkflow \
  --task-queue fishsense_api_queue \
  --input '<dive_id>'
```

**Order matters**: stage 6.1 reads `SpeciesLabel.grouping`. If 6.1
fires before stage 4.2's species sync has touched the dive's labels,
`grouping` is NULL everywhere → each PREDICTION cluster becomes a
single-image LABEL_STUDIO cluster (suboptimal grouping, not broken).
6.1 then refuses to re-run (skip-if-exists, since the cluster API
has no DELETE), so the suboptimal result is locked in. To recover:
manually delete the dive's LABEL_STUDIO clusters in Postgres
(`DELETE FROM dive_frame_cluster WHERE dive_id=? AND data_source='LABEL_STUDIO'`
plus the join table), then re-trigger.

Safe sequence per dive: confirm `species_label.grouping` is
populated for the dive's images (the hourly species sync has run at
least once with completed labels), then trigger 6.1.

#### Cross-worker preprocess parents (stages 0.1, 2, 5.1, 9)

Each preprocess stage splits across both workers: an api-worker
parent does dive selection + SDK resolution + NAS-to-file-exchange
staging, then dispatches a child workflow on
`fishsense_data_processing_queue` for the per-image CPU work. All
four parents are hourly with
`overlap=ScheduleOverlapPolicy.SKIP` and a 15-minute stagger so
their selectors don't all hit `dives.get()` at the top of the hour.

NAS access lives only on the api-worker side
(`stage_raw_bytes_for_dive_activity` for raw `.ORF`s,
`stage_slate_pdf_activity` for stage-9 slate PDFs). Both are
idempotent — a HEAD-check skips already-staged content so retries
are cheap. NAS failure is fatal for the parent run; the next
schedule firing retries from scratch. The data-worker holds no NAS
credentials and only reads/writes the file-exchange.

| Stage | Parent workflow | Schedule offset | Child id pattern |
|---|---|---|---|
| 0.1 | `PreprocessLaserImagesParentWorkflow` | :00 | `preprocess-laser-{dive_id}` |
| 2   | `PreprocessDiveImagesParentWorkflow` | :15 | `preprocess-dive-images-{dive_id}` |
| 5.1 | `PreprocessHeadtailImagesParentWorkflow` | :30 | `preprocess-headtail-{dive_id}` |
| 9   | `PreprocessSlateImagesParentWorkflow` | :45 | `preprocess-slate-{dive_id}` |

Per-stage cohort (selector predicate):

* **0.1** — HIGH-priority dive without `LaserExtrinsics` (matches
  `dry_run_stage13.py`'s target cohort, so preprocessing always
  feeds the next calibration).
* **2** — HIGH-priority dive with PREDICTION clusters AND at least
  one image without a completed `SpeciesLabel`.
* **5.1** — HIGH-priority dive with at least one
  `SpeciesLabel.top_three_photos_of_group=True` whose `HeadTailLabel`
  is missing or incomplete.
* **9** — HIGH-priority dive with `dive_slate_id` set AND at least
  one `SpeciesLabel.content_of_image='Slate, Laser on slate'` whose
  `DiveSlateLabel` is missing or incomplete.

The deterministic child id makes a manual+scheduled trigger overlap
a `WorkflowAlreadyStarted` no-op rather than redoing the dive. Each
parent drains exactly one dive per run, so an N-dive backlog clears
in N hours per stage.

To trigger a manual run for any stage (drains one extra dive
immediately):

```
temporal workflow start \
  --type PreprocessLaserImagesParentWorkflow \
  --task-queue fishsense_api_queue
```

(Same pattern with `Preprocess{DiveImages,Headtail,Slate}ImagesParentWorkflow`
for the others.)

To change cadence / overlap_policy / stagger, `temporal schedule
delete preprocess-{laser,dive-images,headtail,slate}-images-workflow-schedule`
and let the next api-worker restart recreate it.

**Cluster note**: when the data-worker scales to multiple replicas,
per-image activities load-balance for free across replicas (Temporal
task-queue distribution). The parent's `overlap=SKIP` plus
deterministic child ids keep the selector side race-free.

#### Stages 13 + 14 require a data-worker rollout

Stages 13 (laser calibration) and 14 (fish measurement) are baked
into the data-worker image, but the data-worker is held off
auto-deploy (separate host, no in-repo compose). After tagging a
new data-worker version, ops must manually roll it (see "Manual
rollout" above). Without that step:
- Stage 6.1 will create LABEL_STUDIO clusters successfully, but
- Stage 14 measurement won't run, so no `Measurement` rows get
  written and no fish lengths land in the dashboard.

## Where things live in `compose.yml`

```
compose.yml
├── postgres (PG 16, password file via docker secret)
├── qcomm-static-file-server (Traefik + authentik)
├── mafl (homepage at fishsense.e4e.ucsd.edu)
├── include: compose.orchestrator.yml
│     ├── fishsense-api
│     └── static_file_server (nginx DAV — the /api/v1/exchange/ routes)
├── include: compose.superset.yml
├── include: compose.temporal.yml
└── include: compose.workers.yml
      ├── fishsense-api-workflow-worker (× 2 replicas)
      └── fishsense-backup-worker
```

## Related docs

- [CLAUDE.md](../CLAUDE.md) — file-exchange URL contract, build →
  release → promote → deploy pipeline, service Dockerfile pattern,
  worker config-validation gotcha, repo-root settings.toml warning.
- [docs/diagrams.md](../docs/diagrams.md) — system context, deploy
  topology, and per-stage sequence diagrams.

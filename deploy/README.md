# deploy/

Deployment config for FishSense Lite — the prod **Incus slot** interior
(`incus/`), Kubernetes manifests (`k8s/data-worker/`) for the
**data-processing worker**, which runs on Kubernetes rather than a
docker host (NRP/Nautilus is the current target; the Junkyard and
Qualcomm clusters are future ones, not ready yet), and a self-contained
compose stack for the local devcontainer.

## Compose layout

| File | Purpose |
|---|---|
| `incus/compose.yml` | Prod stack, run inside the KRG Incus tenant slot: `traefik` (inner edge), `postgres`, `fishsense-api`, `fishsense-lite-web`, `fishsense-api-workflow-worker`, `fishsense-backup-worker`, the co-located authentik outpost, and Superset behind an off-by-default profile. Converged by `nixos-rebuild`, not by hand — see [`incus/`](incus/README.md). |
| `k8s/data-worker/` | `fishsense-data-processing-workflow-worker` on NRP/Kubernetes (image preprocessing + laser calibration + measurement). **Not** a compose file — `kubectl apply -k k8s/data-worker`; the api-worker scales it to zero when idle. |
| `compose.local.yml` | Self-contained local devcontainer stack — postgres, temporal, fishsense-api (pinned image), Garage (single-node, same engine as prod) + one-shot bootstrap, label-studio, dev container. **Not** layered on the prod stack; prod's authentik / mTLS / letsencrypt coupling is intentionally absent. |

The pre-Incus orchestrator host's compose files (`compose.yml` and its
`compose.orchestrator/temporal/workers/superset.yml` siblings) and their
bind-mount volume dirs were removed once that host was decommissioned —
`git log --diff-filter=D -- deploy/compose.yml` finds them if you need
the history. Temporal now lives on krg-prod (mTLS, `/run/tenant/temporal`),
not in this repo's stack.

`compose.local.yml` and the prod stack are intentionally separate files.
Prod is fronted by Traefik + authentik + letsencrypt + mTLS to Temporal;
trying to make a single set of compose files that boots both on a laptop
and in the slot was messier than splitting them.

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

Two targets: the **Incus slot** runs [`incus/compose.yml`](incus/README.md)
as the tenant interior, and the **data-processing worker** runs as a
Kubernetes Deployment on NRP (see
[`k8s/data-worker/`](k8s/data-worker/README.md)). **Bringing either up
by hand is not the steady-state path** — the auto-deploy pipeline
handles version bumps via reviewable PRs, then
`.github/workflows/deploy.yml` either starts `fishsense-selfupdate` on
the slot's own runner (`nixos-rebuild switch` onto the merged commit) or
runs `kubectl apply -k deploy/k8s/data-worker` from a GitHub-hosted
runner. See the "CI pipeline" section of [CLAUDE.md](../CLAUDE.md).

### Incus slot (prod)

Nothing is checked out onto the slot — `fishsense-selfupdate` runs
`nixos-rebuild switch --flake github:UCSD-E4E/fishsense-lite#fishsense
--refresh`, pulling this repo from GitHub and bringing the compose stack
with it. State lives in named volumes; secrets are rendered by
vault-agent to `/run/tenant/secrets/app.env` (see `incus/secrets.nix`),
so there are no untracked `.secrets/` or `*_volumes/` dirs to restore.

Bootstrap and the remaining operator actions (seed OpenBao, file CNAMEs,
NRP Temporal cert) are in [`incus/README.md`](incus/README.md).

### Data-processing worker (Kubernetes)

The data-worker is CPU-heavy (rectify + JPEG encode + fishsense-core
kernels) and runs on Kubernetes rather than competing with the
orchestrator's postgres / Temporal / authentik. It's a `replicas`-less
Deployment that the api-worker scales 0 ↔ `kubernetes.active_replicas`
on demand. **NRP/Nautilus** is the current target (the Junkyard and
Qualcomm clusters are future ones); all the per-cluster bootstrap (NRP
namespace + permanent-service exception, the three Secrets, the
kubeconfig used by both CI and the api-worker, the api-worker's
`[kubernetes]` config, the orchestrator-side authentik prerequisite) is
in [`k8s/data-worker/README.md`](k8s/data-worker/README.md).

### Rollout operational notes

#### Settings-file changes ride the deploy atomically

`fishsense-selfupdate` rebuilds from the repo at main, so schema changes
to the in-repo `incus/worker_volumes/<svc>/config/settings.toml` files
flow to the slot atomically with the image-pin bump in the same commit —
no drift window where the new image runs against the old settings or
vice versa.

Corollary: a config-only change to `incus/` produces no release and
therefore no `auto-deploy/*` PR, so nothing converges it. Land it on
main, then run `deploy.yml` by hand (`workflow_dispatch`, `target: incus`).

Editing files inside the slot doesn't survive a converge — `nixos-rebuild`
rebuilds the interior from the flake. Always commit config changes.

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
parent does dive selection + SDK resolution + NAS-to-Garage
staging, then dispatches a child workflow on
`fishsense_data_processing_queue` for the per-image CPU work. All
four parents are hourly with
`overlap=ScheduleOverlapPolicy.SKIP` and a 15-minute stagger so
their selectors don't all hit `dives.get()` at the top of the hour.

NAS access lives only on the api-worker side and is **read-only**
(`stage_raw_bytes_for_dive_activity` for raw `.ORF`s,
`stage_slate_pdf_activity` for stage-9 slate PDFs — both download into
the Garage `raw/` / `slate_pdf/` scratch prefixes). Staging is
idempotent — a HeadObject skips already-staged content so retries are
cheap. NAS failure is fatal for the parent run; the next schedule
firing retries from scratch. The data-worker holds no NAS credentials
and only reads/writes the Garage object store (S3).

After the data-worker child completes (it has already written the
processed JPEGs to Garage), the parent runs:

* **`cleanup_raw_bytes_for_dive_activity(dive_id)`** — deletes the
  staged raw `.ORF` *scratch* objects from Garage (`raw/{checksum}.ORF`).
  S3 delete is idempotent. This NEVER touches the NAS source. The
  processed JPEGs stay in Garage (Label Studio reads them via
  presigned URLs) — there is no NAS archive step; Garage is their
  durable home.

| Stage | Parent workflow | Schedule offset | Child id pattern |
|---|---|---|---|
| 0.1 | `PreprocessLaserImagesParentWorkflow` | :00 | `preprocess-laser-{dive_id}` |
| 1   | `ClusterDiveFramesParentWorkflow` | :05 | `cluster-{dive_id}` |
| 2   | `PreprocessSpeciesImagesParentWorkflow` | :15 | `preprocess-species-{dive_id}` |
| 5.1 | `PreprocessHeadtailImagesParentWorkflow` | :30 | `preprocess-headtail-{dive_id}` |
| 9   | `PreprocessSlateImagesParentWorkflow` | :45 | `preprocess-slate-{dive_id}` |

Per-stage cohort (selector predicate):

* **0.1** — HIGH-priority dive with at least one image lacking a
  non-sentinel `LaserLabel` row.
* **1** — HIGH-priority dive with at least one *valid* `LaserLabel`
  (completed, not superseded, x/y both set) AND zero PREDICTION
  `DiveFrameCluster` rows.
* **2** — HIGH-priority dive with PREDICTION clusters AND at least
  one laser-valid image lacking a non-sentinel `SpeciesLabel`.
* **5.1** — HIGH-priority dive with at least one laser-valid image
  lacking a non-sentinel `HeadTailLabel`.
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

**Scale-to-zero on NRP**: each parent above calls
`ensure_data_worker_running_activity` (after it knows there's real
work) to scale the NRP Deployment up to `kubernetes.active_replicas`
before dispatching the child; an hourly `ScaleDownIdleDataWorkerWorkflow`
(slot :55) scales it back to 0 once the data-worker task queue is
quiet. Configured by the api-worker's `[kubernetes]` section
(`kubeconfig_path`, `namespace`, …) — a no-op when that's unset (the
data-worker is then assumed always-on). See [`k8s/data-worker/README.md`](k8s/data-worker/README.md).

#### Stages 13 + 14 ride the data-worker auto-deploy

Stages 13 (laser calibration) and 14 (fish measurement) are baked
into the data-worker image. They roll out via the same auto-deploy
flow as everything else — release-please cuts a data-worker version,
`promote.yml` opens a PR bumping the image `newTag:` in
`k8s/data-worker/kustomization.yaml`, and merging it triggers
`deploy.yml` to `kubectl apply -k deploy/k8s/data-worker` from a
GitHub-hosted runner. No manual step required *once the NRP namespace
+ `NRP_KUBECONFIG` are bootstrapped* (see [`k8s/data-worker/README.md`](k8s/data-worker/README.md));
until then the `deploy-data-worker` job fails fast and stages 13/14
changes won't reach prod.

## Where things live

Incus slot (one runner, auto-registered, labelled `fishsense`):

```
incus/compose.yml                (converged by nixos-rebuild, not by hand)
├── traefik (inner edge :443, fishsense.vm)
├── authentik-outpost (forwardAuth for the API route)
├── postgres (fishsense + superset DBs, vault-agent password)
├── fishsense-lite-web (landing + /portal at fishsense.e4e.ucsd.edu)
├── fishsense-api (api.fishsense.e4e.ucsd.edu, forwardAuth-gated)
├── fishsense-api-workflow-worker  ─┐ krg-prod Temporal over mTLS
├── fishsense-backup-worker        ─┘
└── superset + valkey (profile: superset — off by default)
```

Data-processing worker (NRP/Kubernetes, **not** a compose service):

```
deploy/k8s/data-worker/        (kubectl apply -k)
└── Deployment fishsense-data-processing-workflow-worker
      + ConfigMap (settings.toml), Secrets (creds, Temporal certs, GHCR pull)
      replicas managed by the api-worker (scale 0 <-> active_replicas)
```

## Related docs

- [CLAUDE.md](../CLAUDE.md) — file-exchange URL contract, build →
  release → promote → deploy pipeline, service Dockerfile pattern,
  worker config-validation gotcha, repo-root settings.toml warning.
- [docs/diagrams.md](../docs/diagrams.md) — system context, deploy
  topology, and per-stage sequence diagrams.

# fishsense-lite interior — KRG Incus platform tenant

This directory is the **repo-owned interior** for fishsense as tenant #1 on the
KRG Incus platform (ADR 0017 / 0020). The platform runs this compose stack on
our Incus slot (`10.100.0.10`); `fishsense-selfupdate` converges the instance via
`nixos-rebuild switch --flake github:UCSD-E4E/fishsense-lite#fishsense`, reading
the repo-root [`flake.nix`](../../flake.nix) which imports this directory.

Upstream hand-off:
<https://github.com/KastnerRG/krg-infra/tree/main/docs/handoff/fishsense-lite>

> **Status: live.** The slot is bootstrapped and serving (web + API up). The
> `flake.nix` is at the repo root; `compose.yml` + config + `secrets.nix` +
> `hardware-configuration.nix` live here and are referenced by root-relative paths.
> Three convergence paths: a `promote.yml`-generated `auto-deploy/*` pin-bump PR
> merge (`.github/workflows/deploy.yml`), a manual `workflow_dispatch` (`target: incus`,
> for config-only changes that cut no release), and the **nightly `system.autoUpgrade`**
> (04:00, rebuilds from main — krg-infra #460). The composeStack runs
> `up -d --force-recreate` (krg-infra #459 / our krg-infra#458) so committed config
> changes actually apply on converge. First bring-up was the admin's one-time
> `nixos-rebuild switch`.

## What runs here

| Service | Route | Auth |
|---|---|---|
| `web` (fishsense-lite-web) | `fishsense.e4e.ucsd.edu` | in-app OIDC (`fishsense_oauth`) |
| `fishsense-api` | `api.fishsense.e4e.ucsd.edu` | forwardAuth → co-located outpost |
| `superset` (+init/worker/beat + `valkey`) — **off by default** (`superset` compose profile) | `analytics.fishsense.e4e.ucsd.edu` | in-app OIDC (`fishsense_analytics`) |
| `authentik-outpost` (co-located outpost) | `/outpost.goauthentik.io/*` on api | gates the API route |
| `fishsense-api-workflow-worker` | — | krg-prod Temporal (mTLS) |
| `fishsense-backup-worker` | — | krg-prod Temporal (mTLS) |
| `postgres` (`fishsense` + `superset` DBs) | — | vault-agent password |
| `traefik` (inner edge, `:443`, `fishsense.vm`) | all of the above | — |

**Off-slot (unchanged):** the `fishsense-data-processing-workflow-worker` runs
on NRP/Kubernetes (`deploy/k8s/data-worker/`); the api-worker scales it via the
NRP k8s API. Garage object storage lives on the NAS (reached over S3). Both are
just open-egress destinations from this slot.

> **API hostname rename — cutover consumer.** The API route is
> `api.fishsense.e4e.ucsd.edu` here (was `orchestrator.` on the old host). The
> `fishsense_orchestrator` provider `external_host` is already updated platform-side
> (#437). The one thing still to flip at cutover:
> `deploy/k8s/data-worker/settings.toml` `[fishsense_api].url`. The in-slot web +
> api-worker are unaffected (they use the interior `http://fishsense-api:8000`).

## What changed vs the old orchestrator stack

- **Temporal is external** — krg-prod shared cluster, not in-stack. Dropped the
  `temporal` + `temporal-ui` services, the `workflows.` subdomain, and
  `temporal_db` / `temporal_visibility_db` from Postgres.
- **Routing** flips from the shared host Traefik (docker-label provider +
  letsencrypt + `authentik@docker`) to **our own inner Traefik** with the
  **file provider** (`traefik-dynamic.yml`), serving the vault-agent
  `fishsense.vm` cert on `:443`. No per-service `traefik.*` labels.
- **qcomm docs** static file server dropped (out of tenant scope).

## Temporal (krg-prod)

Endpoint `krg-prod.ucsd.edu:7233` (raw gRPC), namespace `fishsense` (30d
retention), **mTLS-only**, gRPC server-name override → `workflows.krg.ucsd.edu`,
verify against the lab CA. Wired in `worker_volumes/*/config/settings.toml`
`[temporal]`, reading certs from `/run/tenant/temporal/{tls.crt,tls.key,ca.crt}`.

**In-slot: wired by krg-infra PR #435** (ADR 0023). The in-VM vault-agent mints
a `fishsense-worker` client cert to `/run/tenant/temporal/*` (auto-renewed, same
pattern as `fishsense.vm`) — **once we opt in via the `flake.nix` `mkTenant`
`temporal = { namespace = "fishsense" }` field** (no field → no render). The
compose + settings already target these exact paths, so the in-slot workers
connect once the pinned flake (with the opt-in) converges on the slot — #435 and
its OpenBao grant are merged.

**NRP off-prem data-worker: still the manual interim** — an admin hands us a
30-day PEM trio to load as a k8s Secret (automated krg-deploy → NRP-Secret
delivery is a follow-up needing an NRP kubeconfig). Namespace isolation is by
convention, not enforced (krg-infra #434).

Both worker `settings.toml` files point `[temporal]` at krg-prod (api-worker and
backup-worker alike).

## Secrets (HANDOFF §9 — vault-agent renders, `secret/tenants/fishsense/*`)

Nothing secret is committed. `nixosModules.tenant` renders only the **certs**;
`deploy/incus/secrets.nix` (imported by the repo-root `flake.nix`) extends
`krg.vaultAgent.renders` to produce one consolidated
**`/run/tenant/secrets/app.env`**, `env_file`-mounted into the services that need
it. Vault-agent is **fail-closed** — seed every referenced
path before the first converge or the stack won't start.

**Vault-agent render targets:**

| Path | Consumer | Source |
|---|---|---|
| `/run/tenant/tls/fishsense.vm.{crt,key}` | traefik | platform (#428) |
| `/run/tenant/temporal/{tls.crt,tls.key,ca.crt}` | both workers | platform (#435, temporal opt-in) |
| `/run/tenant/secrets/app.env` | postgres, fishsense-api, web, superset×4, both workers | `secrets.nix` |
| `/run/tenant/secrets/backup-postgres.env` | backup-worker (2nd env_file, overrides DB pw → `backup` role) | `secrets.nix` |
| `/run/tenant/outpost/token.env` | authentik-outpost (`AUTHENTIK_TOKEN`; **soft** render, `errorOnMissingKey=false`) | `secrets.nix` |

**DB-role model:** the postgres container + fishsense-api use the **admin** `postgres`
role (`POSTGRES_PASSWORD` / `E4EFS_POSTGRES__PASSWORD` = `postgres.password`); **superset**
uses its own least-priv `superset` role (`DATABASE_PASSWORD` = `superset.db_password`,
read-only SELECT on `fishsense`); the **backup-worker** uses the `backup` role
(`backup_password`, via the override env_file). Three distinct roles.

**OpenBao KV under `secret/tenants/fishsense/` — WE seed:**

| Path | Fields |
|---|---|
| `postgres` | `password`, `backup_password` |
| `superset` | `secret_key`, `db_password` |
| `web` | `auth_secret` |
| `api` | `username`, `password` (fishsense-api basic-auth service acct) |
| `label_studio` | `api_key` |
| `object_store` | `access_key`, `secret_key` (Garage) |
| `nas` | `username`, `password` (FileStation) |

**Platform writes (tofu — do NOT seed):** `oidc/web`, `oidc/analytics`
(`client_id`/`client_secret`/`issuer_url`) (#438), and `oidc/proxy-outpost-token`
(`token` — the co-located outpost's API token, #440).

**Committed non-secret config** (repo binds, self-contained under `deploy/incus/`):
`worker_volumes/*/config/settings.toml`, `fishsense_api_volumes/config/settings.toml`,
`pg_volumes/config/{postgres.conf,pg_hba.conf}`, `superset_volumes/docker/*`
(incl. `superset_config.py` — OIDC derived from the platform `AUTHENTIK_ISSUER`, not
hardcoded). `nrp.kubeconfig` is a vault-agent render / operator-placed file.

**DB bootstrap = restore, not init.** Postgres starts from a **prod `pg_dump` restore**
into the `pgdata` volume (roles + passwords come from the dump); seed OpenBao to match.
`pg_volumes/scripts/` ships no init SQL on purpose — see its README.

## Status — resolved vs. remaining

**Resolved (baked into these files):**
- ✅ `flake.nix` — **at the repo root** (activation done), pinned rev `4c10ed3e`
  (incl. #435–#440, #443, #453, #459 compose force-recreate, #460 nightly auto-upgrade),
  `temporal` opt-in, quota 6/12; imports this dir by root-relative paths.
- ✅ **Disk/boot via `incus-virtual-machine.nix`** (the same module the krg-golden image
  builds from) — systemd-boot + ESP/root fileSystems **by label** + `incus-agent` (keeps
  `incus exec` working post-switch). Replaces the fragile captured-UUID hardware-config;
  the flake also forces OEC off (ephemeral VM tier). *Both belong in `nixosModules.tenant`
  for isVM tenants — flagged upstream; the handoff/template flakes omit them.*
- ✅ `secrets.nix` — §9 app-secret renders (`app.env` + `backup-postgres.env` + soft `token.env`).
- ✅ `.github/workflows/deploy.yml` — the converge job (formerly its own `auto-deploy.yml`)
  now lives in the repo's one deploy workflow, routed by branch name: merging a
  `promote.yml`-generated `auto-deploy/<pkg>-<version>` PR converges the slot; the
  `auto-deploy/fishsense-data-processing-workflow-worker-*` prefix routes to NRP instead.
  A `workflow_dispatch` with `target: incus` re-converges without a pin bump. The
  handoff's `on: push: auto-deploy/**` is deliberately not used — it would fire on the
  *branch push* that opens the PR, converging before a human reviews the pin diff, and on
  the data-worker's branch too. First bring-up doesn't use any of this (admin bootstraps
  manually).
- ✅ **Option A co-located outpost** (`authentik-outpost`, image `2026.5.3` — matches
  krg-prod's Authentik server per krg-infra `compose.authentik.yml`) — HANDOFF §7.
  Platform IaC **landed** (#440): dedicated `fishsense_proxy` outpost, token →
  `oidc/proxy-outpost-token`, soft-rendered (`errorOnMissingKey=false`).
- ✅ API renamed `orchestrator.` → `api.fishsense.e4e.ucsd.edu`; provider `external_host`
  updated platform-side (#437) — matches.
- ✅ Quota 6/12 (#436) + root disk **50 GiB** (#439) landed. Repo public. Data on named volumes.

**Remaining — operator actions (values/creds stay on the box, not in git):**
1. **Seed OpenBao FIRST, then bootstrap.** vault-agent is fail-closed on the *whole*
   agent — a single missing secret blocks even the `fishsense.vm` cert render, so the
   inner Traefik + compose stack won't start (system generation still activates; the
   switch returns non-zero but is recoverable: seed, re-run the switch). Seed **all**
   the `WE seed` KV paths above before the admin's first `nixos-rebuild switch`.
   Do **not** seed `oidc/*` (platform-written; the outpost token renders soft anyway).
2. **Validate the sign-in round-trip** once the API route is up (the outpost image is
   pinned to krg-prod's Authentik `2026.5.3`; bump both together on Authentik upgrades).
3. **File CNAMEs** `api.` + `analytics.` → e4e-prod, confirm they resolve, then let
   #437's SAN re-issue run (staging ACME first, confirm, then off — like the apex).
4. **Apply the quota/disk raise + restart** the instance (6/12 + 50 GiB now merged, #436/#439).
5. **NRP data-worker Temporal cert** (the un-automated piece) — on krg-deploy,
   `bao write pki_int/issue/temporal-client common_name=temporal-worker ttl=720h`,
   load the PEM trio as k8s Secret `temporal-client`, renew at 30 days. Switch CN to
   `fishsense-worker` after #435's grant is live.

**Notes (not blockers):**
- **the converge is a trigger, not a clean CI gate.** `fishsense-selfupdate` stops the
  runner mid-switch, so `deploy.yml` starts it with `systemctl start --no-block` (PR #241)
  and reports success as soon as the converge is *enqueued* — green means "converge
  fired", not "containers healthy". Verify on-box (`systemctl status fishsense-selfupdate`)
  or add a post-converge HTTP health probe.
- **committed config applies only via force-recreate.** The composeStack runs
  `up -d --force-recreate` (krg-infra #459, from our krg-infra#458): plain `up -d` doesn't
  re-read a bind-mounted config file whose store-symlink target changed, so a config edit
  would "deploy" with no effect — it once crash-looped the api-worker on a stale
  `settings.toml`. Trade-off: a *changed* converge recreates the whole stack (brief
  restart of postgres et al.). The **nightly `system.autoUpgrade`** (04:00, #460) means
  any main change also rolls out unattended within a day.
- **Observability — deferred, tracked.** Central tenant metrics are **held** (ours:
  **#220**, upstream krg-infra **#441** — push-based Alloy → mTLS `remote_write` → central
  Grafana, fishsense pilot). No alerting exists yet (ours: **#221**, upstream krg-infra
  **#442**). We expose **no** metrics ports for now; revisit once #441 lands (design any
  in-stack Prometheus `remote_write`-ready so it federates cleanly).

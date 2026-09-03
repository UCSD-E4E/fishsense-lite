# deploy/k8s/data-worker/

Kubernetes manifests for `fishsense-data-processing-workflow-worker`.
This is how the data-worker is deployed — it no longer runs on a
self-hosted docker host via compose. **NRP** (the National Research
Platform / Nautilus, <https://nrp.ai>) is the current target and this
README's bootstrap is NRP-specific (kubeconfig source, the 2-week
Deployment-GC exception, contact-via-Matrix). The **Junkyard** and
**Qualcomm** clusters are longer-term targets — not ready yet; the
Deployment / ConfigMap / Secrets / kustomization here are
cluster-generic and would carry over, only the per-cluster bootstrap
differs.

| File | What |
|---|---|
| `deployment.yaml` | The **CPU** worker (`role=cpu`), serving `fishsense_data_processing_queue`. `replicas` is **omitted** — the api-worker owns the count (scales it up on demand, back to 0 when idle; see below). amd64 nodeSelector, resource requests/limits, `maxSurge: 0`, no PDB, emptyDir scratch, no PVC/GPU/NAS. |
| `deployment-gpu.yaml` | The **GPU** worker (`role=gpu`), serving `fishsense_data_processing_gpu_queue` (laser detector + slate masker). Adds `nvidia.com/gpu: 1`, the `nvidia.com/gpu:NoSchedule` toleration, and SM >= 7.5 node affinity. Allowed to be unschedulable — see "GPU split" below. |
| `deployment-gpu-cpu-fallback.yaml` | The **CPU fallback** for the GPU queue (`role=gpu`, no GPU request). Runs the same checkpoint on the CPU when the GPU worker repeatedly fails to start. Normally 0 replicas, indefinitely. |
| `settings.toml` | Source for the `fishsense-data-worker-settings` ConfigMap (built by kustomize's `configMapGenerator`; mounted at `/e4efs/config/settings.toml`). Credentials are **not** here — they're env vars from a Secret. |
| `kustomization.yaml` | `kubectl apply -k` entrypoint. Holds the overridable image tag (CI bumps it) — one entry covers all three Deployments, which share an image and differ only by `E4EFS_GENERAL__ROLE`. |
| `deployer-rbac.yaml` | The **deploy identity**: ServiceAccount `fishsense-deployer` + a least-privilege Role (deployments/scale/configmaps) + RoleBinding + a token-minting Secret. **Not** in `kustomization.yaml` — operator-applied out-of-band (the SA can't create its own RBAC). Credential-free: the token controller populates the Secret's `.data.token` in-cluster; the JWT never enters git. |

## Who scales it

The data-worker is a **pull-based Temporal worker** — at `replicas: 0`
no pods run, and work just waits on `fishsense_data_processing_queue`
until a worker appears. The **api-worker** (running in the Incus slot,
outside this cluster) brings it back:

* Each parent workflow that dispatches a CPU data-worker child calls
  `ensure_data_worker_running_activity` first — scales `deployment.yaml`
  to `kubernetes.active_replicas` (default 1).
* The two predict parents instead call
  `ensure_gpu_worker_running_activity`, which scales **one of** the GPU
  Deployment or its CPU fallback (see the next section).
* `ScaleDownIdleDataWorkerWorkflow` runs hourly and scales each of the
  three to 0 once the queue *it* serves has had no running or
  recently-closed workflow for `kubernetes.idle_cooldown_minutes`. A busy
  CPU queue therefore never keeps a GPU pod alive.

For that to work the api-worker needs the `e4e-fishsense` kubeconfig
(vault-agent-rendered from OpenBao to `/run/tenant/nrp/kubeconfig`, #245)
and the `[kubernetes]` config section (`kubeconfig_path`, `namespace =
"e4e-fishsense"`, `deployment_name`, `active_replicas`,
`idle_cooldown_minutes`) — see the api-worker's `settings.toml` in
`deploy/incus/worker_volumes/api_worker/config/`. The same kubeconfig is
what CI uses to `kubectl apply` (repo secret `NRP_KUBECONFIG`).

## GPU split, and why nothing waits on a GPU

Until 2026-08-25 there was one Deployment, one queue, and every stage
requested `nvidia.com/gpu: 1` with SM >= 7.5 node affinity. So rectify,
clustering, calibration, measurement and laser depth — none of which touch a
GPU — were all gated on NRP finding a Turing-or-newer card with free capacity,
and then held that card idle through the hours of rawpy decode that dominate a
real dive. At `active_replicas = 2` the second pod sat Pending on
`Insufficient nvidia.com/gpu` (2026-08-20), paying for a card it never used.

Now:

* **`fishsense_data_processing_queue`** — eight stages, no GPU request. Cannot
  be blocked by GPU scarcity.
* **`fishsense_data_processing_gpu_queue`** — the two torch inference stages,
  `predict_laser_image` and (retired) `predict_slate_image`. Served by *either*
  `...-worker-gpu` or `...-worker-gpu-cpu-fallback`, never both.

This queue means **prefer a GPU, not require one** — which is what makes it
safe for a stage that merely benefits from a card. Both detectors run on the
CPU fallback and produce the **same** output there, just slower.

**The fallback state machine.** After `kubernetes.gpu_max_start_failures`
(default 3) consecutive failed GPU starts, the api-worker scales the GPU
Deployment to 0 and the fallback to `kubernetes.gpu_fallback_replicas`, for
`kubernetes.gpu_fallback_minutes` (default 180), then probes the GPU again. A
"failed start" is `spec.replicas > 0` with no Ready pod after
`kubernetes.gpu_start_timeout_seconds` (default 600) — long enough that an
image pull is not mistaken for an outage. The window expires so a transient NRP
shortage cannot strand the stage on CPU inference permanently.

Bookkeeping lives in annotations on the **GPU** Deployment:

```
kubectl -n e4e-fishsense describe deploy fishsense-data-processing-workflow-worker-gpu
```

| Annotation | Meaning |
|---|---|
| `fishsense.e4e.ucsd.edu/gpu-start-failures` | consecutive failed starts |
| `fishsense.e4e.ucsd.edu/gpu-wedged-since` | when the current wedge was first seen |
| `fishsense.e4e.ucsd.edu/gpu-fallback-until` | CPU fallback holds until this time |

**All three are absent when everything is healthy**, so seeing any of them is
itself the signal. They are operator-editable:

```bash
# Force CPU inference now (e.g. you know the GPU pool is drained):
kubectl -n e4e-fishsense annotate deploy fishsense-data-processing-workflow-worker-gpu \
  fishsense.e4e.ucsd.edu/gpu-start-failures=3 --overwrite

# End a fallback early and retry the GPU on the next firing:
kubectl -n e4e-fishsense annotate deploy fishsense-data-processing-workflow-worker-gpu \
  fishsense.e4e.ucsd.edu/gpu-fallback-until- fishsense.e4e.ucsd.edu/gpu-start-failures-
```

`kubectl apply` does not clear them — apply only prunes fields it previously
managed, and these are written by the api-worker.

**A red GPU rollout in CI is a warning, not a failure.** `deploy.yml` rolls out
all three but only warns if the GPU one does not converge, precisely so a GPU
shortage on NRP cannot block shipping the CPU stages.

## Laser auto-accept — the switch and the dark run

The gate that decides which laser predictions skip human review is configured
entirely from settings, so the two operations that matter most need no deploy.
Every key has a default; none is required.

| Key | Default | Meaning |
|---|---|---|
| `laser_auto_accept.enabled` | `true` | Off = kill switch **and** dark run |
| `laser_auto_accept.audit_sample_rate` | `0.10` | Share of cleared frames sent to a human anyway |
| `laser_auto_accept.min_predictions` | `20` | Frames a dive needs before any may be cleared |
| `laser_auto_accept.min_inlier_fraction` | `0.75` | How much of the dive must agree on one line |
| `laser_auto_accept.max_perpendicular_px` | `10.0` | Distance from that line a cleared dot may sit |
| `laser_auto_accept.max_along_line_z` | `4.0` | How far along the line, as a robust z |

**`enabled = false` does not stop the gate — it stops it *acting*.** The fit
still runs, and every verdict and margin is still written to
`LaserPrediction`. So a week with it off measures exactly what the gate would
have done to real dives, on current data, with no frame skipping a person.
That makes it the right first move on a new environment, and the right move if
something looks wrong later: nothing has to be unwound, because the verdict is
advisory until populate reads `auto_accept`.

Expect rows reading `auto_accept = false` beside `gate_verdict =
'auto_accepted'` while it is off. That pair is the switch overruling the fit,
not a bug.

```bash
# Stop it acting, keep measuring:
kubectl -n e4e-fishsense set env deploy/fishsense-data-processing-workflow-worker \
  E4EFS_LASER_AUTO_ACCEPT__ENABLED=false

# Softer version — clears frames but sends every one to a human anyway:
kubectl -n e4e-fishsense set env deploy/fishsense-data-processing-workflow-worker \
  E4EFS_LASER_AUTO_ACCEPT__AUDIT_SAMPLE_RATE=1.0
```

**What to watch is the per-dive verdict mix**, logged by the predict parent on
every firing. It is free, it is per dive, and it needs no human labels — where
the audit sample is slow and a poor instrument for rare events. Alert on
**both** tails. A dive routing far more frames to people than the ~13% pool
baseline is a detector or an environment that changed. A suspiciously *low*
flag rate in a new environment is the signature of the one failure per-dive
consensus cannot self-detect: a majority of predictions wrong in a
mutually-consistent way, with the true dots flagged as the minority.

## One-time bootstrap (per NRP namespace)

1. Namespace: **`e4e-fishsense`** (already provisioned; pinned in
   `kustomization.yaml`). If standing up a fresh one, request the
   namespace + enough quota from NRP support for
   `active_replicas × the Deployment's limits`, **and** ask (in
   Matrix, per <https://nrp.ai/contact>) for a permanent-service
   exception for the namespace. NRP garbage-collects Deployments
   older than 2 weeks unless the namespace is on the exceptions list
   — this one's permanent. (ConfigMaps/Secrets aren't subject to that
   policy, and our pods are owned by a ReplicaSet so the 6-hour
   bare-pod rule doesn't apply either; the Deployment itself is the
   only thing that needs the exception.)
2. Build a **static-token kubeconfig** for `e4e-fishsense`. NRP's own
   kubeconfig uses interactive `kubelogin`/OIDC (CILogon browser flow),
   which can't run in CI or the api-worker — so both non-interactive
   consumers authenticate as the `fishsense-deployer` ServiceAccount
   instead. One human, once, with an interactive kubeconfig:

   ```sh
   # a. Create the deploy identity (SA + Role + RoleBinding + token Secret).
   #    Declarative + credential-free; idempotent (adopts anything you
   #    created imperatively).
   kubectl apply -f deploy/k8s/data-worker/deployer-rbac.yaml

   # b. Read back the token + CA the token controller populated. If TOKEN
   #    is empty after ~10s, NRP policy stripped the long-lived Secret —
   #    fall back to: kubectl -n e4e-fishsense create token \
   #      fishsense-deployer --duration=168h  (expires; needs re-mint).
   TOKEN=$(kubectl -n e4e-fishsense get secret fishsense-deployer-token \
     -o jsonpath='{.data.token}' | base64 -d)
   CA=$(kubectl -n e4e-fishsense get secret fishsense-deployer-token \
     -o jsonpath='{.data.ca\.crt}')            # already base64
   SERVER=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')

   # c. Assemble a kubeconfig with NO exec block (that is what makes it
   #    non-interactive). This file DOES carry the token — keep it out of
   #    git; it only goes into the two secret stores in step (d).
   cat > nrp.kubeconfig <<KUBECONFIG
   apiVersion: v1
   kind: Config
   clusters: [{name: nrp, cluster: {server: $SERVER, certificate-authority-data: $CA}}]
   users: [{name: fishsense-deployer, user: {token: $TOKEN}}]
   contexts: [{name: fishsense, context: {cluster: nrp, user: fishsense-deployer, namespace: e4e-fishsense}}]
   current-context: fishsense
   KUBECONFIG

   # d. Verify it in isolation, then seed both consumers.
   KUBECONFIG=./nrp.kubeconfig kubectl -n e4e-fishsense auth can-i patch deployments/scale  # -> yes
   ```

   The verified `nrp.kubeconfig` is used in two places — the repo secret
   `NRP_KUBECONFIG` (CI deploys) and the api-worker on the Incus slot
   (scaling). The slot's copy is **not** a mounted file: seed it into
   OpenBao at `secret/tenants/fishsense/nrp { kubeconfig }` and
   vault-agent renders it to `/run/tenant/nrp/kubeconfig` in the slot
   (see `deploy/incus/secrets.nix` + the api-worker's
   `[kubernetes].kubeconfig_path`; wired by #245). A long-lived
   SA-token Secret does not expire, but a **bound** token (the fallback)
   does — on renewal, reseed OpenBao + rotate `NRP_KUBECONFIG`.
3. Create the two Secrets the Deployment references. (No image pull
   secret — the GHCR image is public, so the Deployment has no
   `imagePullSecrets` and kubelet pulls it anonymously.)

   ```sh
   # (all in the e4e-fishsense namespace — the kubeconfig context or -n)

   # 1. Service-account creds (SDK HTTP Basic — the password is the authentik
   #    app-password from `secret/tenants/fishsense/oidc/data-worker-apppw`,
   #    NOT the raw AD password; see krg-infra #484) + Garage S3 key/secret.
   kubectl create secret generic fishsense-data-worker-secrets -n e4e-fishsense \
     --from-literal=fishsense_api_username='<svc>' \
     --from-literal=fishsense_api_password='<app-password>' \
     --from-literal=object_store_access_key='<garage-access-key>' \
     --from-literal=object_store_secret_key='<garage-secret-key>'

   # 2. Temporal mTLS material — the krg-prod client identity. This is the
   #    SAME identity the slot's workers use, CN=fishsense-worker, not a
   #    second one: krg-infra's tenant module issues
   #    `pki_int/issue/temporal-client common_name=<tenant>-worker`.
   #    Bootstrap only — see "Cert rotation" below; after the first apply
   #    the slot re-pushes this Secret on every rotation. Mint on krg-deploy:
   #      bao write pki_int/issue/temporal-client \
   #        common_name=fishsense-worker ttl=720h
   #    (CN authorized for the fishsense tenant now that krg-infra #435's
   #    grant is live). root-ca.pem is krg-prod's Temporal CA (the KRG Lab
   #    Internal Intermediate, good to 2031) — only the leaf turns over.
   kubectl create secret generic fishsense-data-worker-temporal-certs -n e4e-fishsense \
     --from-file=client.pem=/path/to/fishsense-data-processing-workflow-worker.pem \
     --from-file=client.key=/path/to/fishsense-data-processing-workflow-worker.key \
     --from-file=root-ca.pem=/path/to/root-ca.pem
   ```

   (Credentials in git are out of scope here — if you want them
   reconciled by `kubectl apply` instead, add Sealed Secrets / SOPS.)
4. First apply manually: `kubectl apply -k deploy/k8s/data-worker`
   (the namespace `e4e-fishsense` is pinned in `kustomization.yaml`, so
   no `-n` needed). The Deployment manifest omits `replicas`
   (the api-worker owns it), so this first create comes up at the k8s
   default of 1 — `kubectl scale deployment/fishsense-data-processing-workflow-worker --replicas=0`
   right after, or just leave it: the hourly idle-sweeper scales it to
   0 within the hour. After that, rollouts ride CI: `promote.yml`
   bumps the image tag in `kustomization.yaml` and opens an
   `auto-deploy/fishsense-data-processing-workflow-worker-*` PR;
   merging it triggers `deploy.yml` to `kubectl apply -k` again.

## Cert rotation — the slot pushes it, nothing here pulls it

The krg-prod Temporal leaf is a **7-day** cert. On the Incus slot,
vault-agent re-renders it before expiry (`krg.vaultAgent.renewal`,
krg-infra #534) and the root `flake.nix`'s `temporal.reload` restarts the
services holding it — re-rendering alone recovers nothing, because a
worker builds its TLS config once at `Client.connect` and keeps the old
leaf for the life of the process.

vault-agent cannot reach into NRP, so this Secret was originally minted by
hand at `ttl=720h` and renewed by nobody. **It expired 2026-08-14 05:46:26
UTC**, every pod went `CrashLoopBackOff` on `received fatal alert:
CertificateExpired`, and the v2.15.2 rollout on 2026-08-18 timed out with
no stated cause — which read as NRP having killed the deploy. (NRP had in
fact GC'd the Deployment separately; `apply` recreated it fine. The cert is
what kept it down.)

That gap is closed by
[`deploy/incus/nrp_cert_sync/`](../../incus/nrp_cert_sync/): a one-shot
compose service on the slot, listed in `flake.nix`'s `temporal.reload`,
that pushes the rotated leaf into this Secret and `rollout restart`s the
Deployment. It is a no-op when the leaf is unchanged (it compares a
`fishsense.e4e.ucsd.edu/leaf-sha256` annotation), so it is safe on every
converge. `deploy.yml` also refuses to deploy on an expired leaf, naming
it in seconds rather than timing out after five minutes.

Consequences worth knowing:

* **The Secret is no longer hand-managed.** Editing it by hand is fine for
  an emergency, but the next rotation overwrites it. Fix the forwarder,
  not the Secret.
* **Do not `kubectl delete` this Secret casually.** The forwarder recreates
  it on the next rotation, which can be days away; until then every pod
  crash-loops. Re-run the forwarder explicitly instead:
  `docker compose ... restart nrp-temporal-cert-sync` on the slot.
* The leaf the data-worker now carries is a 7-day cert refreshed roughly
  every 5 days, not the old 30-day one. That is strictly better — but it
  means a forwarder that silently stops shows up within a week.

## NRP's 2-week Deployment GC

The bootstrap above asks NRP for a permanent-service exception. **As of
2026-08-20 the namespace does not appear to hold one** — the Deployment's
`creationTimestamp` was 2026-08-17 while its ConfigMaps still dated to
July, i.e. the object had been collected and recreated by a deploy.
ConfigMaps and Secrets are not time-GC'd, and our pods are ReplicaSet-owned
so the 6-hour bare-pod rule never applies; the Deployment is the only thing
at risk.

This is survivable — `kubectl apply -k` recreates it, and `deploy.yml`
logs a `::warning::` when it has to, so repeated collection is visible in
the job log rather than silent. Two caveats: a recreated Deployment starts
at `replicas: 1` instead of whatever the api-worker had scaled it to, and
any work queued while it was absent simply waits. Still worth chasing the
exception with NRP support (Matrix, <https://nrp.ai/contact>).

## Prerequisite on the Incus slot side

The data-worker is Garage-only: it reaches the hosted Garage (S3) object
store directly at its public endpoint with S3 access keys, talks to the
public `api.fishsense.e4e.ucsd.edu` route, and polls the shared **krg-prod**
Temporal cluster (`settings.toml` `[temporal]` → `krg-prod.ucsd.edu`,
matching the api-worker — the two share `fishsense_data_processing_queue`,
so a mismatched cluster means dispatched children time out forever). Slot /
platform-side TODO:

- **Seed OpenBao `secret/tenants/fishsense/nrp { kubeconfig }`** with the
  `e4e-fishsense` token kubeconfig, so vault-agent renders it into the slot
  and the api-worker can scale this Deployment (#245). Do this **before**
  the api-worker converge that enables `[kubernetes].kubeconfig_path`, or it
  crash-loops on the missing file (isolated to that worker).
- Mint a Garage S3 access key scoped to read the `raw/` + `slate_pdf/`
  scratch prefixes and write the JPEG prefixes; put it in the
  `fishsense-data-worker-secrets` Secret above.
- Garage must send CORS headers for the labeler origin (Label Studio
  presigns the JPEGs and the browser fetches them directly) — but
  that's a Label-Studio-serving concern, not a worker one.
- The SDK presents `fishsense_api.username/password` to
  `api.fishsense.e4e.ucsd.edu` through authentik basic-auth-passthrough;
  reuse the existing service account. Confirm the passthrough policy
  covers the service account on the API paths the SDK hits.
- Temporal reachability is to krg-prod (`:7233`) from NRP node ranges —
  a krg-prod ingress concern, not the (now-decommissioned) orchestrator's.

## Not here

No PVC (stateless), no GPU (fishsense-core is CPU/Rust), no NAS mount
(the api-worker stages all bytes into the Garage object store before
dispatching) — and no in-cluster autoscaler (KEDA/CronJob): scaling is
the api-worker's job, deliberately, so the control plane stays on
hardware we own and NRP is pure burst compute.

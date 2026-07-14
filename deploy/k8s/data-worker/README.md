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
| `deployment.yaml` | The Deployment. `replicas` is **omitted** — the api-worker owns the count (scales it up on demand, back to 0 when idle; see below). amd64 nodeSelector, resource requests/limits, `maxSurge: 0`, no PDB, emptyDir scratch, no PVC/GPU/NAS. |
| `settings.toml` | Source for the `fishsense-data-worker-settings` ConfigMap (built by kustomize's `configMapGenerator`; mounted at `/e4efs/config/settings.toml`). Credentials are **not** here — they're env vars from a Secret. |
| `kustomization.yaml` | `kubectl apply -k` entrypoint. Holds the overridable image tag (CI bumps it). |

## Who scales it

The data-worker is a **pull-based Temporal worker** — at `replicas: 0`
no pods run, and work just waits on `fishsense_data_processing_queue`
until a worker appears. The **api-worker** (running in the Incus slot,
outside this cluster) brings it back:

* Each parent workflow that dispatches a data-worker child calls
  `ensure_data_worker_running_activity` first — scales this Deployment
  to `kubernetes.active_replicas` (default 1).
* `ScaleDownIdleDataWorkerWorkflow` runs hourly and scales it to 0 once
  the data-worker task queue has had no running or recently-closed
  workflow for `kubernetes.idle_cooldown_minutes`.

For that to work the api-worker needs the `e4e-fishsense` kubeconfig
(vault-agent-rendered from OpenBao to `/run/tenant/nrp/kubeconfig`, #245)
and the `[kubernetes]` config section (`kubeconfig_path`, `namespace =
"e4e-fishsense"`, `deployment_name`, `active_replicas`,
`idle_cooldown_minutes`) — see the api-worker's `settings.toml` in
`deploy/incus/worker_volumes/api_worker/config/`. The same kubeconfig is
what CI uses to `kubectl apply` (repo secret `NRP_KUBECONFIG`).

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
2. Get a (token) kubeconfig for `e4e-fishsense`. It's used in two
   places — the repo secret `NRP_KUBECONFIG` (CI deploys) and the
   api-worker on the Incus slot (scaling). The slot's copy is **not**
   a mounted file: seed it into OpenBao at
   `secret/tenants/fishsense/nrp { kubeconfig }` and vault-agent renders
   it to `/run/tenant/nrp/kubeconfig` in the slot (see
   `deploy/incus/secrets.nix` + the api-worker's `[kubernetes].kubeconfig_path`;
   wired by #245). The token **expires** — reseed OpenBao + rotate
   `NRP_KUBECONFIG` on renewal.
3. Create the three Secrets the Deployment references:

   ```sh
   # (all in the e4e-fishsense namespace — the kubeconfig context or -n)

   # 1. Service-account creds (SDK HTTP Basic, via authentik passthrough)
   #    + Garage S3 access key/secret for the object store
   kubectl create secret generic fishsense-data-worker-secrets -n e4e-fishsense \
     --from-literal=fishsense_api_username='<svc>' \
     --from-literal=fishsense_api_password='<svc-pw>' \
     --from-literal=object_store_access_key='<garage-access-key>' \
     --from-literal=object_store_secret_key='<garage-secret-key>'

   # 2. Temporal mTLS material — the data-worker's OWN krg-prod client
   #    identity, distinct from the api-worker's. Mint on krg-deploy:
   #      bao write pki_int/issue/temporal-client \
   #        common_name=fishsense-worker ttl=720h
   #    (CN authorized for the fishsense tenant now that krg-infra #435's
   #    grant is live). Renew ~30d. root-ca.pem is krg-prod's Temporal CA.
   kubectl create secret generic fishsense-data-worker-temporal-certs -n e4e-fishsense \
     --from-file=client.pem=/path/to/fishsense-data-processing-workflow-worker.pem \
     --from-file=client.key=/path/to/fishsense-data-processing-workflow-worker.key \
     --from-file=root-ca.pem=/path/to/root-ca.pem

   # 3. Pull secret for the private GHCR image
   kubectl create secret docker-registry ghcr-pull -n e4e-fishsense \
     --docker-server=ghcr.io \
     --docker-username='<github-user>' \
     --docker-password='<PAT with read:packages on UCSD-E4E>'
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

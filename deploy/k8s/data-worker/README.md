# deploy/k8s/data-worker/

Kubernetes manifests for `fishsense-data-processing-workflow-worker`
running on **NRP** (the National Research Platform / Nautilus,
<https://nrp.ai>). Replaces `deploy/compose.data-worker.yml` — the
data-worker no longer runs on a self-hosted docker host.

| File | What |
|---|---|
| `deployment.yaml` | The Deployment. `replicas: 0` baseline — the api-worker scales it up on demand and back to 0 when idle (see below). amd64 nodeSelector, resource requests/limits, `maxSurge: 0`, no PDB, emptyDir scratch, no PVC/GPU/NAS. |
| `settings.toml` | Source for the `fishsense-data-worker-settings` ConfigMap (built by kustomize's `configMapGenerator`; mounted at `/e4efs/config/settings.toml`). Credentials are **not** here — they're env vars from a Secret. |
| `kustomization.yaml` | `kubectl apply -k` entrypoint. Holds the overridable image tag (CI bumps it). |

## Who scales it

The data-worker is a **pull-based Temporal worker** — at `replicas: 0`
no pods run, and work just waits on `fishsense_data_processing_queue`
until a worker appears. The **api-worker** (running on the orchestrator,
outside this cluster) brings it back:

* Each parent workflow that dispatches a data-worker child calls
  `ensure_data_worker_running_activity` first — scales this Deployment
  to `kubernetes.active_replicas` (default 1).
* `ScaleDownIdleDataWorkerWorkflow` runs hourly and scales it to 0 once
  the data-worker task queue has had no running or recently-closed
  workflow for `kubernetes.idle_cooldown_minutes`.

For that to work the api-worker needs an NRP kubeconfig and the
`[kubernetes]` config section (`kubeconfig_path`, `namespace`,
`deployment_name`, `active_replicas`, `idle_cooldown_minutes`) — see
the api-worker's `settings.toml` / `deploy/README.md`. The same
kubeconfig is what CI uses to `kubectl apply` (repo secret
`NRP_KUBECONFIG`).

## One-time bootstrap (per NRP namespace)

1. Request a namespace + enough quota from NRP support for
   `active_replicas × the Deployment's limits`.
2. Get a (token) kubeconfig for that namespace. It's used in two
   places — the repo secret `NRP_KUBECONFIG` (CI deploys) and the
   api-worker's mounted kubeconfig (scaling) — and **expires**, so put
   a renewal reminder somewhere.
3. Create the three Secrets the Deployment references:

   ```sh
   # 1. Service-account creds (HTTP Basic, via authentik passthrough)
   kubectl create secret generic fishsense-data-worker-secrets \
     --from-literal=fishsense_api_username='<svc>' \
     --from-literal=fishsense_api_password='<svc-pw>' \
     --from-literal=file_exchange_username='<exchange-svc>' \
     --from-literal=file_exchange_password='<exchange-svc-pw>'

   # 2. Temporal mTLS material (the data-worker's OWN client cert/key +
   #    the same root CA the other workers use)
   kubectl create secret generic fishsense-data-worker-temporal-certs \
     --from-file=client.pem=/path/to/fishsense-data-processing-workflow-worker.pem \
     --from-file=client.key=/path/to/fishsense-data-processing-workflow-worker.key \
     --from-file=root-ca.pem=/path/to/root-ca.pem

   # 3. Pull secret for the private GHCR image
   kubectl create secret docker-registry ghcr-pull \
     --docker-server=ghcr.io \
     --docker-username='<github-user>' \
     --docker-password='<PAT with read:packages on UCSD-E4E>'
   ```

   (Credentials in git are out of scope here — if you want them
   reconciled by `kubectl apply` instead, add Sealed Secrets / SOPS.)
4. First apply manually: `kubectl apply -k deploy/k8s/data-worker`
   (set the namespace via `-n` / the kubeconfig context, or pin it in
   `kustomization.yaml`). After that, rollouts ride CI: `promote.yml`
   bumps the image tag in `kustomization.yaml` and opens an
   `auto-deploy/fishsense-data-processing-workflow-worker-*` PR;
   merging it triggers `deploy.yml` to `kubectl apply -k` again.

## Prerequisite on the orchestrator side

The `/api/v1/exchange/*` Traefik route must be moved from its
single-IP allowlist to authentik basic-auth-passthrough before the
NRP worker can reach the file-exchange (NRP pods have no stable egress
IP). See `deploy/compose.orchestrator.yml` and the project notes. Also
confirm the orchestrator's firewall allows Temporal `:7233` ingress
from NRP node ranges.

## Not here

No PVC (stateless), no GPU (fishsense-core is CPU/Rust), no NAS mount
(the api-worker stages all bytes onto the file-exchange before
dispatching) — and no in-cluster autoscaler (KEDA/CronJob): scaling is
the api-worker's job, deliberately, so the control plane stays on
hardware we own and NRP is pure burst compute.

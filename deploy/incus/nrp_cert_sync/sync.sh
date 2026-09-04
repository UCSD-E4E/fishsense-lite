#!/bin/sh
# Mirror the slot's rotated Temporal leaf into the NRP data-worker's Secret.
#
# WHY THIS EXISTS. The krg-prod Temporal mTLS leaf is a 7-day cert. On the slot,
# vault-agent re-renders it before expiry (krg.vaultAgent.renewal, krg-infra #534)
# and `temporal.reload` in flake.nix restarts the compose services that hold it —
# a process builds its TLS config once at `Client.connect`, so a fresh file on
# /run recovers nothing on its own.
#
# The data-worker runs on NRP, outside the slot, and vault-agent cannot reach it.
# Its copy of the SAME identity lived in a hand-minted `ttl=720h` k8s Secret that
# nothing renewed. It expired 2026-08-14 05:46:26 UTC and every pod went
# CrashLoopBackOff on `received fatal alert: CertificateExpired`, which is what
# made the v2.15.2 rollout time out. The 2026-08-17 rotation fix covered the two
# slot services and missed this one. This script closes that gap: the slot is
# where the renewed cert lands, so the slot is what pushes it onward.
#
# Same identity, not a second one: krg-infra's tenant module issues
# `pki_int/issue/temporal-client common_name=<tenant>-worker` -> CN=fishsense-worker,
# which is exactly the CN the expired k8s cert carried. `ca.crt` is the render of
# `.Data.issuing_ca` (CN=KRG Lab Internal Intermediate CA), matching root-ca.pem.
#
# Run as a one-shot compose service (`restart: "no"`), listed in flake.nix's
# `temporal.reload`. That hook is `docker compose ... restart <svc>...`, which
# starts an exited container again — so each rotation re-runs this. It also runs
# on every converge, which is harmless: it is a no-op when the leaf is unchanged.
#
# Paths are overridable so CI can exercise this against a kind cluster
# (.github/workflows/k8s-tests.yml).
set -eu

TEMPORAL_CERT_DIR="${TEMPORAL_CERT_DIR:-/run/tenant/temporal}"
KUBECONFIG="${KUBECONFIG:-/run/tenant/nrp/kubeconfig}"
NRP_NAMESPACE="${NRP_NAMESPACE:-e4e-fishsense}"
SECRET_NAME="${SECRET_NAME:-fishsense-data-worker-temporal-certs}"
# EVERY Deployment that mounts $SECRET_NAME must be rolled, not just the first.
# The data-worker is three Deployments as of the GPU/CPU split (cpu, gpu, and
# the gpu CPU-fallback) and all three mount this same leaf. Rolling only one
# would leave the others holding an expired cert and crash-looping on
# `CertificateExpired` — which is precisely the 2026-08-14 outage this whole
# forwarder exists to prevent, reintroduced through the back door.
# Space-separated and overridable so CI can point it at one name.
DEPLOY_NAME="${DEPLOY_NAME:-fishsense-data-processing-workflow-worker}"
DEPLOY_NAMES="${DEPLOY_NAMES:-${DEPLOY_NAME} ${DEPLOY_NAME}-gpu ${DEPLOY_NAME}-gpu-cpu-fallback ${DEPLOY_NAME}-light}"
# Records which leaf the Secret currently holds, so a converge that changed
# nothing does not roll the data-worker.
FP_ANNOTATION="fishsense.e4e.ucsd.edu/leaf-sha256"
export KUBECONFIG

log() { echo "[nrp-temporal-cert-sync] $*"; }

# The NRP kubeconfig is a SOFT vault-agent render — it can legitimately be
# absent (not yet seeded, or a slot brought up without NRP). That is not a
# failure of the interior stack, so exit clean and say why.
if [ ! -r "$KUBECONFIG" ]; then
    log "no kubeconfig at $KUBECONFIG - nothing to sync (soft render absent)"
    exit 0
fi

for f in tls.crt tls.key ca.crt; do
    if [ ! -r "$TEMPORAL_CERT_DIR/$f" ]; then
        log "ERROR: missing $TEMPORAL_CERT_DIR/$f - is the temporal render enabled?"
        exit 1
    fi
done

fingerprint="$(sha256sum "$TEMPORAL_CERT_DIR/tls.crt" | cut -d' ' -f1)"

# `|| true`: a missing Secret (first run, or one wiped alongside the namespace)
# must fall through to the create below, not abort under `set -e`.
#
# The key is spelled out again here rather than interpolating $FP_ANNOTATION:
# kubectl's jsonpath needs every dot in an annotation KEY backslash-escaped, so
# the two spellings genuinely differ. Keep them in step.
current="$(kubectl -n "$NRP_NAMESPACE" get secret "$SECRET_NAME" \
    -o "jsonpath={.metadata.annotations['fishsense\\.e4e\\.ucsd\\.edu/leaf-sha256']}" \
    2>/dev/null || true)"

if [ -n "$current" ] && [ "$current" = "$fingerprint" ]; then
    log "leaf unchanged ($fingerprint) - no update, no rollout"
    exit 0
fi

log "pushing rotated leaf to $NRP_NAMESPACE/$SECRET_NAME"
# create --dry-run | apply is the idempotent upsert: it creates the Secret when
# absent and replaces the three keys when present, without a delete window.
kubectl -n "$NRP_NAMESPACE" create secret generic "$SECRET_NAME" \
    --from-file=client.pem="$TEMPORAL_CERT_DIR/tls.crt" \
    --from-file=client.key="$TEMPORAL_CERT_DIR/tls.key" \
    --from-file=root-ca.pem="$TEMPORAL_CERT_DIR/ca.crt" \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl -n "$NRP_NAMESPACE" annotate secret "$SECRET_NAME" \
    "$FP_ANNOTATION=$fingerprint" --overwrite

# kubelet refreshes a mounted Secret in place, but the worker reads /certs once
# at Client.connect - so running pods keep the old leaf until they are replaced.
# A no-op when the api-worker has it scaled to 0: the annotation lands on the
# pod template and the next scale-up starts on the new cert.
for deploy in $DEPLOY_NAMES; do
    # A Deployment that does not exist is skipped rather than fatal: the GPU
    # pair only exists once the split's manifests have been applied, and a
    # rotation must not start failing on a cluster that is mid-upgrade.
    if ! kubectl -n "$NRP_NAMESPACE" get "deployment/$deploy" >/dev/null 2>&1; then
        log "$deploy not present - skipping"
        continue
    fi
    log "rolling $deploy onto the new leaf"
    # kubectl refuses a second `rollout restart` inside the same second ("if
    # restart has already been triggered within the past second"). That guard
    # firing means a restart just landed, which is the outcome we wanted - take
    # it rather than failing the sync under `set -e`. Any other failure is real
    # and propagates.
    if restart_out="$(kubectl -n "$NRP_NAMESPACE" rollout restart "deployment/$deploy" 2>&1)"; then
        log "$restart_out"
    else
        case "$restart_out" in
            *"within the past second"*)
                log "a rollout was already triggered this second - taking it" ;;
            *)
                log "ERROR: $restart_out"
                exit 1 ;;
        esac
    fi
done

log "done ($fingerprint)"

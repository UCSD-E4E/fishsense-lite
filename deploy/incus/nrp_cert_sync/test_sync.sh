#!/usr/bin/env bash
# Exercises sync.sh against a real cluster (kind in CI, see
# .github/workflows/k8s-tests.yml). Needs a working KUBECONFIG, the
# `e4e-fishsense` namespace, and the data-worker Deployment applied.
#
# sync.sh never parses the cert bodies — it hashes tls.crt and uploads all
# three files — so plain text fixtures cover the real behaviour and the test
# needs no PKI tooling.
set -euo pipefail

NS="${NRP_NAMESPACE:-e4e-fishsense}"
SECRET=fishsense-data-worker-temporal-certs
DEPLOY=fishsense-data-processing-workflow-worker
# Every Deployment that mounts the leaf. All three must roll on a rotation, or
# the ones left behind hold an expired cert and crash-loop on
# `CertificateExpired` - the 2026-08-14 outage, reintroduced.
DEPLOYS=("$DEPLOY" "$DEPLOY-gpu" "$DEPLOY-gpu-cpu-fallback" "$DEPLOY-light")
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
CERTS="$WORK/temporal"
mkdir -p "$CERTS"

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "  ok - $*"; }

restarted_at() {
    kubectl -n "$NS" get "deployment/${1:-$DEPLOY}" \
        -o 'jsonpath={.spec.template.metadata.annotations.kubectl\.kubernetes\.io/restartedAt}' 2>/dev/null || true
}
# Concatenated restart stamps for every Deployment, so a check covers the whole
# set rather than whichever one happens to be first.
all_restarted_at() {
    local d
    for d in "${DEPLOYS[@]}"; do printf '%s=%s;' "$d" "$(restarted_at "$d")"; done
}
leaf_annotation() {
    kubectl -n "$NS" get secret "$SECRET" \
        -o "jsonpath={.metadata.annotations['fishsense\\.e4e\\.ucsd\\.edu/leaf-sha256']}" 2>/dev/null || true
}
run_sync() { TEMPORAL_CERT_DIR="$CERTS" NRP_NAMESPACE="$NS" "$HERE/sync.sh"; }

echo "1. absent kubeconfig is a clean no-op, not a stack failure"
out="$(TEMPORAL_CERT_DIR="$CERTS" KUBECONFIG="$WORK/nope" "$HERE/sync.sh")" \
    || fail "expected exit 0 when the soft render is absent"
grep -q "nothing to sync" <<<"$out" || fail "expected a 'nothing to sync' message, got: $out"
pass "exits 0 and explains itself"

echo "2. a missing cert render is a hard error"
if TEMPORAL_CERT_DIR="$CERTS" NRP_NAMESPACE="$NS" "$HERE/sync.sh" >/dev/null 2>&1; then
    fail "expected non-zero exit when tls.crt is missing"
fi
pass "exits non-zero"

printf 'leaf-v1\n' > "$CERTS/tls.crt"
printf 'key-v1\n'  > "$CERTS/tls.key"
printf 'ca-v1\n'   > "$CERTS/ca.crt"

echo "3. first run pushes the leaf and rolls every deployment"
before_roll="$(all_restarted_at)"
run_sync >/dev/null
[[ "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.client\.pem}' | base64 -d)" == "leaf-v1" ]] \
    || fail "client.pem was not updated from tls.crt"
[[ "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.client\.key}' | base64 -d)" == "key-v1" ]] \
    || fail "client.key was not updated from tls.key"
[[ "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.root-ca\.pem}' | base64 -d)" == "ca-v1" ]] \
    || fail "root-ca.pem was not updated from ca.crt"
expected="$(sha256sum "$CERTS/tls.crt" | cut -d' ' -f1)"
[[ "$(leaf_annotation)" == "$expected" ]] || fail "leaf-sha256 annotation not recorded"
after_roll="$(all_restarted_at)"
[[ "$after_roll" != "$before_roll" ]] || fail "deployments were not rolled"
# Assert EACH one individually - a concatenated compare would pass while some
# were skipped, which is exactly the failure mode being guarded against.
for d in "${DEPLOYS[@]}"; do
    [[ -n "$(restarted_at "$d")" ]] || fail "$d was not rolled onto the new leaf"
done
pass "secret upserted, annotated, all ${#DEPLOYS[@]} deployments rolled"

echo "4. an unchanged leaf does not roll the deployments again"
sleep 1   # restartedAt is second-resolution; without this a real roll could alias
before_roll="$(all_restarted_at)"
out="$(run_sync)"
grep -q "leaf unchanged" <<<"$out" || fail "expected the unchanged short-circuit, got: $out"
[[ "$(all_restarted_at)" == "$before_roll" ]] || fail "deployments rolled on an unchanged leaf"
pass "no-op on a converge that rotated nothing"

echo "5. a rotated leaf is pushed and rolls every deployment"
printf 'leaf-v2\n' > "$CERTS/tls.crt"
printf 'key-v2\n'  > "$CERTS/tls.key"
sleep 1
before_roll="$(all_restarted_at)"
declare -A before_each
for d in "${DEPLOYS[@]}"; do before_each["$d"]="$(restarted_at "$d")"; done
run_sync >/dev/null
[[ "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.client\.pem}' | base64 -d)" == "leaf-v2" ]] \
    || fail "rotated client.pem was not pushed"
[[ "$(leaf_annotation)" == "$(sha256sum "$CERTS/tls.crt" | cut -d' ' -f1)" ]] \
    || fail "leaf-sha256 annotation not updated on rotation"
[[ "$(all_restarted_at)" != "$before_roll" ]] || fail "deployments were not rolled on rotation"
for d in "${DEPLOYS[@]}"; do
    [[ "$(restarted_at "$d")" != "${before_each[$d]}" ]] \
        || fail "$d was not rolled on rotation - it would keep the expired leaf"
done
pass "rotation propagates and rolls all ${#DEPLOYS[@]}"

echo "6. it recreates the Secret if it is missing entirely"
kubectl -n "$NS" delete secret "$SECRET" >/dev/null
run_sync >/dev/null
[[ "$(kubectl -n "$NS" get secret "$SECRET" -o jsonpath='{.data.client\.pem}' | base64 -d)" == "leaf-v2" ]] \
    || fail "Secret was not recreated from scratch"
pass "recreates a deleted Secret"

echo "ALL PASS"

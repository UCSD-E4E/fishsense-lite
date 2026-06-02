#!/bin/sh
# Idempotent single-node Garage bootstrap for the local/CI stack:
# assign a storage layout, create the `fishsense` bucket, import a FIXED
# access key (so tests have deterministic creds), and grant it rw.
# Safe to re-run — each step is guarded.
set -eu

echo "garage-init: waiting for daemon..."
until garage status >/dev/null 2>&1; do sleep 1; done

if garage status 2>/dev/null | grep -q "NO ROLE ASSIGNED"; then
    node=$(garage node id -q | cut -d@ -f1)
    echo "garage-init: assigning layout to ${node}"
    garage layout assign -z dc1 -c 1G "${node}"
    garage layout apply --version 1
else
    echo "garage-init: layout already assigned; skipping"
fi

garage bucket create "${OBJECT_STORE_BUCKET}" 2>/dev/null \
    && echo "garage-init: bucket ${OBJECT_STORE_BUCKET} created" \
    || echo "garage-init: bucket ${OBJECT_STORE_BUCKET} already exists"

if garage key list 2>/dev/null | grep -q "${OBJECT_STORE_ACCESS_KEY}"; then
    echo "garage-init: key already imported; skipping"
else
    echo "garage-init: importing fixed access key"
    garage key import --yes -n fishsense-key \
        "${OBJECT_STORE_ACCESS_KEY}" "${OBJECT_STORE_SECRET_KEY}"
fi

garage bucket allow --read --write --owner \
    "${OBJECT_STORE_BUCKET}" --key fishsense-key
echo "garage-init: done"

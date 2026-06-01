#!/bin/sh
# Render the live Garage config from the committed (secret-free) template:
# prepend a randomly-generated rpc_secret as a TOP-LEVEL key, then the
# template body. Runs once per fresh `garage_config` volume; idempotent.
#
# Why generate instead of committing the secret: garage v1.0.1's `server`
# requires rpc_secret to be present in the config file (the GARAGE_RPC_SECRET
# env is honored by the CLI client but not the server's config loader), so
# we render it at runtime onto a private volume rather than checking a
# secret-shaped string into git. The value protects only this single-node
# local cluster and never leaves the volume.
#
# NOTE: rpc_secret MUST be prepended (before any `[section]` header) so TOML
# scopes it at the top level — appending it after `[s3_api]` would make it
# `s3_api.rpc_secret` and the daemon would report it missing.
set -eu

config=/config/garage.toml

if [ -s "${config}" ]; then
    echo "garage-keygen: config already rendered; skipping"
    exit 0
fi

secret=$(head -c 32 /dev/urandom | od -An -t x1 | tr -d ' \n')
{
    printf 'rpc_secret = "%s"\n\n' "${secret}"
    cat /template
} > "${config}"
echo "garage-keygen: rendered ${config} with a freshly-generated rpc_secret"

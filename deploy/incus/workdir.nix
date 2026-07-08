# Populate the composeStack working directory.
#
# The krg composeStack runner (krg-infra services/compose-stack.nix) invokes our
# compose with `--project-directory /var/lib/krg/fishsense`, so docker resolves
# the compose file's RELATIVE bind/env_file paths (`./superset_volumes/docker/.env`,
# `./traefik-dynamic.yml`, `./pg_volumes/config`, …) against THAT dir — its doc
# says "relative paths in compose files resolve here" — NOT the Nix-store compose
# dir. That dir is otherwise empty, so `docker compose up` fails
# (`env file .../superset_volumes/docker/.env not found`).
#
# Fix: symlink the repo-committed READ-ONLY config into the working dir, pointing
# at the flake's store copy (the store path changes on every config edit, so
# `L+` refreshes the link each converge — repo-owns-deploy preserved). The
# READ-WRITE worker log dirs are made as real directories, not store symlinks.
#
# Paths (`./x`) are relative to this file (deploy/incus/), so each resolves to
# that subtree's Nix store path.
{
  systemd.tmpfiles.rules = [
    # read-only config (mounted :ro in compose)
    "L+ /var/lib/krg/fishsense/traefik-dynamic.yml    - - - - ${./traefik-dynamic.yml}"
    "L+ /var/lib/krg/fishsense/superset_volumes       - - - - ${./superset_volumes}"
    "L+ /var/lib/krg/fishsense/pg_volumes             - - - - ${./pg_volumes}"
    "L+ /var/lib/krg/fishsense/fishsense_api_volumes  - - - - ${./fishsense_api_volumes}"
    # worker: config is read-only (symlink), logs are read-write (real dir)
    "d  /var/lib/krg/fishsense/worker_volumes                       0755 root root -"
    "d  /var/lib/krg/fishsense/worker_volumes/api_worker            0755 root root -"
    "L+ /var/lib/krg/fishsense/worker_volumes/api_worker/config     - - - - ${./worker_volumes/api_worker/config}"
    "d  /var/lib/krg/fishsense/worker_volumes/api_worker/logs       0755 root root -"
    "d  /var/lib/krg/fishsense/worker_volumes/backup_worker         0755 root root -"
    "L+ /var/lib/krg/fishsense/worker_volumes/backup_worker/config  - - - - ${./worker_volumes/backup_worker/config}"
    "d  /var/lib/krg/fishsense/worker_volumes/backup_worker/logs    0755 root root -"
  ];
}

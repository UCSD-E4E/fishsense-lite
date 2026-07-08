# App-secret delivery for the fishsense interior (HANDOFF §9).
#
# nixosModules.tenant's vault-agent renders only the CERTS (fishsense.vm,
# temporal). App secrets need their own render, which we add here — the list
# merges with the platform's. The tenant AppRole reads
# secret/data/tenants/fishsense/*, so it can render everything under there.
#
# One consolidated env file is rendered to a tmpfs path and env_file-mounted
# into the services that need it (compose.yml). Vars a service doesn't recognize
# are ignored, so a single file is simplest.
#
# ── OpenBao KV layout under secret/tenants/fishsense/ (KV-v2) ───────────────────
# WE seed (operator — see README):
#   postgres      { password, backup_password }  # admin pw + the `backup` role's pw
#   superset      { secret_key, db_password }     # flask secret + the `superset` DB role's pw
#   web           { auth_secret }              # next-auth cookie/JWT signing
#   api           { username, password }       # fishsense-api basic-auth service acct
#   label_studio  { api_key }
#   object_store  { access_key, secret_key }   # Garage S3
#   nas           { username, password }        # Synology FileStation
# PLATFORM writes (tofu — do NOT seed):
#   oidc/web                 { client_id, client_secret, issuer_url }   (#438)
#   oidc/analytics           { client_id, client_secret, issuer_url }   (#438)
#   oidc/proxy-outpost-token { token }   (co-located outpost API token — #440)
#
# ⚠️ vault-agent is FAIL-CLOSED (errorOnMissingKey): a referenced path/field that
# isn't seeded takes the whole stack down rather than starting empty. Seed every
# path below before the first converge.
{
  krg.vaultAgent.renders = [
    {
      # Shared secrets, env_file-mounted into postgres, web, superset×4, and both
      # workers. E4EFS_POSTGRES__PASSWORD here is the ADMIN role (fishsense-api +
      # container); the backup-worker overrides it with a second env_file below.
      destination = "/run/tenant/secrets/app.env";
      contents = ''
        {{ with secret "secret/data/tenants/fishsense/postgres" }}POSTGRES_PASSWORD={{ .Data.data.password }}
        E4EFS_POSTGRES__PASSWORD={{ .Data.data.password }}{{ end }}
        {{/* superset connects as the least-priv `superset` role — DATABASE_PASSWORD is
             that role's password, NOT admin (superset_config.py DATABASE_USER=superset) */}}
        {{ with secret "secret/data/tenants/fishsense/superset" }}SUPERSET_SECRET_KEY={{ .Data.data.secret_key }}
        DATABASE_PASSWORD={{ .Data.data.db_password }}{{ end }}

        {{/* fishsense-lite-web (next-auth + landing) */}}
        {{ with secret "secret/data/tenants/fishsense/web" }}AUTH_SECRET={{ .Data.data.auth_secret }}{{ end }}
        {{ with secret "secret/data/tenants/fishsense/api" }}FISHSENSE_API_USERNAME={{ .Data.data.username }}
        FISHSENSE_API_PASSWORD={{ .Data.data.password }}{{ end }}
        {{ with secret "secret/data/tenants/fishsense/label_studio" }}LABEL_STUDIO_API_KEY={{ .Data.data.api_key }}{{ end }}
        {{ with secret "secret/data/tenants/fishsense/oidc/web" }}AUTH_AUTHENTIK_ID={{ .Data.data.client_id }}
        AUTH_AUTHENTIK_SECRET={{ .Data.data.client_secret }}
        AUTH_AUTHENTIK_ISSUER={{ .Data.data.issuer_url }}{{ end }}

        {{/* superset OIDC (analytics) — names read by superset_config.py OAUTH_PROVIDERS */}}
        {{ with secret "secret/data/tenants/fishsense/oidc/analytics" }}AUTHENTIK_KEY={{ .Data.data.client_id }}
        AUTHENTIK_SECRET={{ .Data.data.client_secret }}
        AUTHENTIK_ISSUER={{ .Data.data.issuer_url }}{{ end }}

        {{/* api-worker + backup-worker — dynaconf E4EFS_<SECTION>__<KEY> env override */}}
        {{ with secret "secret/data/tenants/fishsense/api" }}E4EFS_FISHSENSE_API__USERNAME={{ .Data.data.username }}
        E4EFS_FISHSENSE_API__PASSWORD={{ .Data.data.password }}{{ end }}
        {{ with secret "secret/data/tenants/fishsense/object_store" }}E4EFS_OBJECT_STORE__ACCESS_KEY={{ .Data.data.access_key }}
        E4EFS_OBJECT_STORE__SECRET_KEY={{ .Data.data.secret_key }}{{ end }}
        {{ with secret "secret/data/tenants/fishsense/label_studio" }}E4EFS_LABEL_STUDIO__API_KEY={{ .Data.data.api_key }}{{ end }}
        {{ with secret "secret/data/tenants/fishsense/nas" }}E4EFS_E4E_NAS__USERNAME={{ .Data.data.username }}
        E4EFS_E4E_NAS__PASSWORD={{ .Data.data.password }}{{ end }}
      '';
    }
    {
      # Backup-worker override: it connects to postgres as the least-priv `backup`
      # role, not admin. Mounted as the SECOND env_file on that service so this
      # E4EFS_POSTGRES__PASSWORD wins over app.env's admin value.
      destination = "/run/tenant/secrets/backup-postgres.env";
      contents = ''
        {{ with secret "secret/data/tenants/fishsense/postgres" }}E4EFS_POSTGRES__PASSWORD={{ .Data.data.backup_password }}{{ end }}
      '';
    }
    {
      # Co-located Authentik outpost API token (platform-written by #440, only after
      # the terraform/authentik apply). SOFT render — errorOnMissingKey=false so an
      # un-minted token can't fail-close the WHOLE agent (which would also block the
      # fishsense.vm cert → inner Traefik). Missing token = outpost can't connect
      # (login fails, loud but localized) while the rest of the stack stays up.
      destination = "/run/tenant/outpost/token.env";
      errorOnMissingKey = false;
      contents = ''
        {{ with secret "secret/data/tenants/fishsense/oidc/proxy-outpost-token" }}AUTHENTIK_TOKEN={{ .Data.data.token }}{{ end }}
      '';
    }
  ];
}

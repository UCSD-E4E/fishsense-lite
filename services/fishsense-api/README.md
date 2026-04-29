# fishsense-api

FastAPI app that fronts the FishSense Postgres database. Single source
of truth for cameras, dives, dive slates, fish, images, labels, and
users; the workers in this repo call it via
`fishsense-api-sdk`.

## Layout

```
src/fishsense_api/
  server.py        # FastAPI app + lifespan that wires up the engine
  database.py      # Async SQLAlchemy engine + SQLModel.metadata.create_all
  config.py        # Dynaconf settings (postgres.*)
  controllers/     # Route handlers, one module per resource
  models/          # SQLModel tables — authoritative schema
  alembic/         # Migration scripts
```

The `controllers/` and `clients/` directories in
[fishsense-api-sdk](../../libs/fishsense-api-sdk/) are paired one-for-one
with the SQLModel resources here; when you add a resource on this
side, add the matching SDK client and Pydantic model.

## Required env (`E4EFS_` prefix)

- `E4EFS_POSTGRES__HOST`, `E4EFS_POSTGRES__PORT`, `E4EFS_POSTGRES__USERNAME`, `E4EFS_POSTGRES__PASSWORD`, `E4EFS_POSTGRES__DATABASE`

The Dynaconf validators in `config.py` enforce these at first
`settings` access, so missing env values fail at startup with a clear
error rather than at first DB call.

## Running

In the local devcontainer the api is brought up automatically by
`deploy/compose.local.yml`. To run manually outside the container:

```
uv run --package fishsense-api uvicorn fishsense_api:app --reload
```

Docs are at `/docs`. The lifespan calls `SQLModel.metadata.create_all`
at startup — fine for the local stack, but in prod schema changes go
through alembic.

## Migrations (alembic)

`alembic.ini` lives at the package root; the script directory is
`src/fishsense_api/alembic/`. Generate a new migration with
`uv run --package fishsense-api alembic revision --autogenerate -m "<msg>"`
and apply with `... alembic upgrade head`. Migrations are checked into
`src/fishsense_api/alembic/versions/`.

## Production caveats

- `fishsense-api` is **production-only** — there is no separate test environment, and DB backups (via `fishsense-backup-worker`) are the rollback path. Treat schema changes carefully and prefer read-only validation against existing prod values where possible.
- The public host (`orchestrator.fishsense.e4e.ucsd.edu`) is behind Authentik OAuth; SDK basic-auth requests will be 302'd. Internal callers (workers in the same docker network) hit the in-cluster service name and skip the proxy.

## Tests

```
./check.sh unit           # default markers (skips integration)
./check.sh integration    # requires the local stack
```

`tests/test_sdk_drift.py` is the SDK ↔ API model mirror check —
it imports both sides (the SDK is a dev-only dependency on this
package) and diffs field sets. Four `label_studio_json` fields are
allowlisted as known divergence.

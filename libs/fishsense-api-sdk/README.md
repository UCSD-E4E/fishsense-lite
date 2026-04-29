# fishsense-api-sdk

Async Python HTTP client for `fishsense-api`. Used by both Temporal
workers in this repo (`fishsense-api-workflow-worker`,
`fishsense-data-processing-workflow-worker`) and by external notebooks /
scripts.

## Usage

```python
from fishsense_api_sdk import Client

async with Client(base_url, username, password) as client:
    cameras = await client.cameras.list()
    image = await client.images.get(checksum)
    await client.labels.create_laser_label(...)
```

The single `Client` is a façade; each resource kind has its own
sub-client (`cameras`, `dives`, `dive_slates`, `fish`, `images`,
`labels`, `users`) wired up in `client.py`. They share an
`asyncio.Semaphore` so `max_concurrent_requests` caps concurrency
across resources, not per-resource.

`async with` is required — each sub-client owns an `httpx.AsyncClient`
that's opened in `__aenter__` and closed in `__aexit__`.

## Authentication

Currently HTTP basic auth via `username` / `password`. Note:
`orchestrator.fishsense.e4e.ucsd.edu` is fronted by Authentik OAuth in
prod, which 302-redirects basic-auth requests — for dev access use
Postgres direct or a port-forward to the internal API instead of going
through the public host.

## Models

`fishsense_api_sdk.models.*` are Pydantic models that **hand-mirror**
the SQLModel table definitions in `services/fishsense-api/src/fishsense_api/models/`.
There is no codegen; a drift test
(`services/fishsense-api/tests/test_sdk_drift.py`) compares field sets
between the two sides on every PR. Four `label_studio_json` fields are
allowlisted as known divergence — see the test for details.

When you change a SQLModel field on the API side, mirror the change
here in the same PR (or update the allowlist with reasoning).

## Versioning / publishing

Versioned by `python-semantic-release` from conventional commits. The
SDK is **not** published as an image (it's a library); release-please
cuts a tag, the wheel is built and uploaded by CI, and consumers pin
via `fishsense-api-sdk = { workspace = true }` inside this monorepo or
by version outside it.

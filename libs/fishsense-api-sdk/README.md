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

HTTP basic auth via `username` / `password`.

The public host is `api.fishsense.e4e.ucsd.edu` (renamed from `orchestrator.`
at the Incus migration) and sits behind an Authentik forwardAuth middleware. A
**302 from it means the credentials were rejected**, not that basic auth is
unsupported — passthrough is enabled. Workers are unaffected either way: they
reach `fishsense-api:8000` on the interior docker network and never touch the
proxy.

For dev access from outside, easiest first: `incus exec` into the slot and point
`fishsense_api.url` at the interior address; or read Postgres directly for
read-only work, which has no auth story and no write surface to worry about.

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

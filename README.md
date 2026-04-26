# fishsense-lite

UCSD E4E FishSense Lite system monorepo. Internal services consolidated from
the previous polyrepo split:

- `services/fishsense-api/` — FastAPI server in front of PostgreSQL
- `services/fishsense-api-workflow-worker/` — Temporal worker for Label Studio sync
- `services/fishsense-data-processing-workflow-worker/` — Temporal worker for image processing
- `deploy/` — docker compose stack (ex `fishsense-web-services`)
- `libs/fishsense-shared/` — shared Dynaconf / TLS / logging helpers

External repos consumed via git refs (not part of this monorepo):

- [fishsense-core](https://github.com/UCSD-E4E/fishsense-core) — Rust + PyO3 compute library
- [fishsense-api-sdk](https://github.com/UCSD-E4E/fishsense-api-sdk) — Python HTTP client SDK

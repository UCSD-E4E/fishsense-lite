# fishsense-lite-mono — Claude operating notes

Loose ends and architectural conventions that aren't otherwise tracked.

## Notebook port status

| Stage | Notebook | Owner | Status |
|---|---|---|---|
| 0.1 | preprocess_laser_images | data-worker | ported |
| 0.3 | populate_label_studio_project | api-worker | not started |
| 1   | cluster_dive_frames | data-worker | ported (pre-existing) |
| 2   | preprocess_dive_images | data-worker | ported |
| 4   | populate_label_studio_project | api-worker | not started |
| 4.2 | sync_species_labels | api-worker | not started |
| 5.1 | preprocess_headtail_images | data-worker | ported |
| 5.3 | populate_label_studio_project | api-worker | not started |
| 6.1 | update_dive_image_groups | api-worker | not started |
| 9   | preprocess_slate_images | data-worker | ported |
| 11  | populate_label_studio_project | api-worker | not started |
| 12  | sync_slate_label | api-worker | not started |
| 13  | perform_laser_calibration | api-worker | not started (kernel in fishsense-core) |
| 14  | measure_fish | api-worker | not started (kernel in fishsense-core) |

Owner column is the *target* worker once the api-worker side is built;
currently the api-worker has no activities yet, so all api-worker
notebooks are still hand-run.

## Data-worker activity pattern

Every ported per-image stage follows the same shape:

1. `download_raw(checksum)` from the file-exchange.
2. (stage 9 only) `download_slate_pdf(slate_id)`.
3. Off-loop CPU work via `asyncio.to_thread`: rectify
   (`RectifiedImage(RawImage(bytes), intrinsics)` — rawpy + auto-gamma +
   CLAHE + `cv2.undistort`) → stage-specific overlay → `cv2.imencode`.
4. `upload_processed_jpeg(folder, checksum, jpeg_bytes)` to the
   file-exchange.

Output folders match the labeler-facing GET routes already in
`deploy/static_file_server/nginx.conf`:

| Stage | Folder |
|---|---|
| 2   | `preprocess_groups_jpeg` |
| 0.1 | `preprocess_jpeg` |
| 5.1 | `preprocess_headtail_jpeg` |
| 9   | `preprocess_slate_images_jpeg` |

Each port has the same 4-test TDD structure: pure-logic overlay/encode
unit tests, in-process Temporal workflow contract test, integration
test against real `.ORF` fixture (`-m integration`), notebook byte-parity
test (`-m integration`). The integration + parity tests share the same
`tests/fixtures/stage2_sample.ORF` — there's no per-stage raw fixture.

Stage 5.1's parity test also doubles as a proof that
`RawImage(p).data → cv2.undistort(...)` equals
`RectifiedImage(RawImage(p), intrinsics).data` byte-for-byte.

The stages are *intentionally not refactored* into a shared base
activity. Each has a distinct overlay shape (text vs rectangle vs
PDF-composite) and a distinct DTO; one shared signature would have to
be `Callable[[ndarray], ndarray]` plus union-typed extra args, which
is messier than four small, self-contained activities.

## File-exchange URL contract

```
GET  /api/v1/exchange/raw/{checksum}.ORF             # api-worker stages, data-worker reads
GET  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf # api-worker stages, data-worker reads (stage 9)
PUT  /api/v1/exchange/{folder}/{checksum}.JPG        # data-worker writes
```

The nginx DAV alias at `/api/v1/exchange/` covers any subpath, so
adding new conventions is a `FileExchangeClient` change only — no
nginx.conf change needed.

## Worker config validation gotcha

Dynaconf eagerly validates **every** `Validator` on first attribute
access of `settings`, not lazily per setting. Tests that import any
activity module must plumb env values for all required settings
(`temporal.host`, `e4e_nas.url`, `fishsense_api.url`, etc.) even if
the test only uses one of them — see `configure_worker_settings` in
`test_stage2_integration.py` for the standard placeholder fixture.

The `*.url` validators use a custom `_url_condition` (http/https +
non-empty hostname) instead of `validators.url`, because the strict
library condition rejects every Docker-internal hostname
(`static_file_server`, `fishsense-api`, `temporal` — underscores or no
TLD). Don't switch back to `validators.url`.

## Repo-root `settings.toml` — do NOT commit

`fishsense_shared.get_config_path()` falls back to `cwd` outside Docker,
so the worker reads `./settings.toml` when run from the repo root.
Running it locally creates this file as a side-effect; it has prod-y
URLs inside.

Polyrepo `settings.toml` leftovers were intentionally cleaned up in
`6c3920b` and the same file coming back from local-running is the same
problem. Leave it untracked. If a committed file is genuinely needed,
the right shape is `settings.toml.example` + gitignore (matches the
`deploy/.env.local.example` pattern).

## `fishsense-core` 1.7.0 → 1.7.1 was bundled into the stage-2 port

Bumped in working tree before the stage-2 commit and rolled into commit
`669f933` rather than split out. If you're tracing why a particular
fishsense-core version is in `uv.lock`, look at `669f933` *and* the
workspace pyproject change in `75d2979` (the prior 1.7.0 bump).

## Other open work lives in project memory

Migration findings #1 (core↔sdk dep direction) and #4 (service
Dockerfiles broken in monorepo layout), the four `label_studio_json`
SDK drift allowlist entries, the stage14 sign flip, and the phase-6
cutover items are all in
`~/.claude/projects/.../memory/`. Start there for status of in-flight
work.

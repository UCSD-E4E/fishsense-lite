# fishsense-data-processing-workflow-worker

Temporal worker for per-image preprocessing — rectify the raw `.ORF`,
draw a stage-specific overlay, JPEG-encode, and `PUT` the result back
to the file-exchange. CPU-bound, opencv-heavy.

Task queue: `fishsense_data_processing_queue`.

Intended deploy targets: Waiter, the Junkyard / Qualcomm clusters, or
Nautilus — separate hosts from the orchestrator that runs the api +
api-worker, since this image is heavy (opencv, rawpy, scikit-image,
fishsense-core).

## Workflows

| Stage | Workflow | Output folder | Status |
|---|---|---|---|
| —     | `DiveFrameClusteringWorkflow`       | n/a (DB writes)              | always-on |
| 2     | `PreprocessDiveImagesWorkflow`      | `preprocess_groups_jpeg`     | flag-gated |
| 0.1   | `PreprocessLaserImagesWorkflow`     | `preprocess_jpeg`            | flag-gated |
| 5.1   | `PreprocessHeadtailImagesWorkflow`  | `preprocess_headtail_jpeg`   | flag-gated |
| 9     | `PreprocessSlateImagesWorkflow`     | `preprocess_slate_images_jpeg` | flag-gated |

The four flag-gated workflows are registered with the worker only when
`E4EFS_FEATURE_FLAGS__NEW_PREPROCESS_WORKFLOWS=true`. Default OFF —
deploying the binary without that env set means only the legacy
`DiveFrameClusteringWorkflow` runs. Workflow tasks for the gated types
will sit in the queue forever (no worker claims them) rather than
erroring, so the gate is a soft block.

Lift the flag once the api-worker driver for a given stage exists *and*
the relevant math has been re-verified on real frames (especially the
stage 14 sign concern). The gate is deliberately all-or-nothing — if
finer control is needed, split into per-stage flags rather than gating
inside workflow code (workflow code must stay deterministic).

## Activity pattern

Every per-image stage follows the same shape:

1. `download_raw(checksum)` from the file-exchange.
2. (stage 9 only) `download_slate_pdf(slate_id)`.
3. Off-loop CPU work via `asyncio.to_thread`: rectify
   (`RectifiedImage(RawImage(bytes), intrinsics)` — rawpy + auto-gamma
   + CLAHE + `cv2.undistort`) → stage-specific overlay → `cv2.imencode`.
4. `upload_processed_jpeg(folder, checksum, jpeg_bytes)` to the
   file-exchange.

The stages are intentionally **not** refactored into a shared base
activity. Each has a distinct overlay shape (text vs rectangle vs
PDF-composite) and a distinct DTO; one shared signature would have to
be `Callable[[ndarray], ndarray]` plus union-typed extras, which is
messier than four small self-contained activities.

## File-exchange contract

```
GET  /api/v1/exchange/raw/{checksum}.ORF             # this worker reads
GET  /api/v1/exchange/dive_slate_pdfs/{slate_id}.pdf # stage 9 only
PUT  /api/v1/exchange/{folder}/{checksum}.JPG        # this worker writes
```

`{folder}` is one of the four output folders in the table above;
labeler-facing GET routes for these are configured in
[deploy/static_file_server/nginx.conf](../../deploy/static_file_server/nginx.conf).

## Required env (`E4EFS_` prefix)

```
E4EFS_TEMPORAL__HOST, E4EFS_TEMPORAL__PORT
E4EFS_TEMPORAL__TLS=true|false
E4EFS_TEMPORAL__CLIENT_CERT, E4EFS_TEMPORAL__CLIENT_PRIVATE_KEY  # when tls=true
E4EFS_E4E_NAS__URL, E4EFS_E4E_NAS__USERNAME, E4EFS_E4E_NAS__PASSWORD
E4EFS_FISHSENSE_API__URL
E4EFS_STATIC_FILE_SERVER__URL
E4EFS_FEATURE_FLAGS__NEW_PREPROCESS_WORKFLOWS=true|false  # default false
```

The `*.url` validators use a custom `_url_condition` (http/https +
non-empty hostname) instead of `validators.url`, because the strict
library condition rejects every Docker-internal hostname
(`static_file_server`, `fishsense-api`, `temporal` — underscores or no
TLD). Don't switch back to `validators.url`.

The envvar prefix is **`E4EFS_`** (was `DYNACONF_` in the polyrepo —
hosts running an older version of this worker must rename their env
vars before deploying this one).

## Tests

```
./check.sh unit           # default markers (skips integration)
./check.sh integration    # needs the local devcontainer stack
```

Each port has the same 4-test TDD structure: pure-logic
overlay/encode unit tests, in-process Temporal workflow contract
test, integration test against a real `.ORF` fixture (`-m
integration`), and a notebook byte-parity test (`-m integration`).
The integration + parity tests share `tests/fixtures/stage2_sample.ORF`
— there is no per-stage raw fixture.

Integration tests pass workflows directly to a one-off `Worker(...)`
and don't depend on the `feature_flags.new_preprocess_workflows` gate,
so the flag has no effect on tests.

## Running

```
uv run --package fishsense-data-processing-workflow-worker \
    fishsense_data_processing_workflow_worker
```

The image is **not** auto-deployed by the
[deploy.yml](../../.github/workflows/deploy.yml) workflow — its host
isn't in this repo's compose files. `:v<version>` tags are still
published by `promote.yml`, so manual rollout on the data-worker host
is `docker compose pull && up -d`.

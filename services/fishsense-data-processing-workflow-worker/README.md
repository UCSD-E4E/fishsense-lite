# fishsense-data-processing-workflow-worker

Temporal worker for per-image preprocessing — rectify the raw `.ORF`, draw a
stage-specific overlay, JPEG-encode, and write the result to the Garage object
store. CPU-bound, opencv-heavy.

It is **Garage-only**: it reads the raw scratch the api-worker staged and writes
JPEGs, and holds no NAS credentials at all. That is what lets it run off-prem
(NRP) without a stable egress IP — S3 access keys authenticate from anywhere,
whereas the NAS does not — and it keeps NAS credentials off the cluster.

Task queue: `fishsense_data_processing_queue`.

Intended deploy targets: Waiter, the Junkyard / Qualcomm clusters, or
Nautilus — separate hosts from the orchestrator that runs the api +
api-worker, since this image is heavy (opencv, rawpy, scikit-image,
fishsense-core).

## Workflows

| Stage | Workflow | Output folder |
|---|---|---|
| 1     | `DiveFrameClusteringWorkflow`       | n/a — returns `list[list[int]]` (image_ids per cluster); api-worker parent persists PREDICTION clusters via SDK |
| 2     | `PreprocessSpeciesImagesWorkflow`   | `preprocess_groups_jpeg` (started as child by `PreprocessSpeciesImagesParentWorkflow` on the api-worker) |
| 0.1   | `PreprocessLaserImagesWorkflow`     | `preprocess_jpeg` (started as child by `PreprocessLaserImagesParentWorkflow` on the api-worker) |
| 5.1   | `PreprocessHeadtailImagesWorkflow`  | `preprocess_headtail_jpeg` (started as child by `PreprocessHeadtailImagesParentWorkflow` on the api-worker) |
| 9     | `PreprocessSlateImagesWorkflow`     | `preprocess_slate_images_jpeg` (started as child by `PreprocessSlateImagesParentWorkflow` on the api-worker) |

## Activity pattern

Every per-image stage follows the same shape:

1. `download_raw(checksum)` from Garage (`raw/{checksum}.ORF`).
2. (stage 9 only) `download_slate_pdf(slate_id)`
   (`slate_pdf/{slate_id}.pdf`).
3. Off-loop CPU work via `asyncio.to_thread`: rectify
   (`RectifiedImage(RawImage(bytes), intrinsics)` — rawpy + auto-gamma
   + CLAHE + `cv2.undistort`) → stage-specific overlay → `cv2.imencode`.
4. `upload_processed_jpeg(folder, checksum, jpeg_bytes)` to Garage
   (`{jpeg_prefix}/{checksum}.JPG`). Those keys are **physical** — the populate
   activities embed the same `s3://` URIs in Label Studio task data, so there is
   no virtual→physical rewrite layer. The key helpers live once, in
   `fishsense_shared.object_store`.

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
E4EFS_FISHSENSE_API__URL
E4EFS_FILE_EXCHANGE__URL
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

## Running

```
uv run --package fishsense-data-processing-workflow-worker \
    fishsense_data_processing_workflow_worker
```

Auto-deployed by the
[deploy.yml](../../.github/workflows/deploy.yml) data-worker job: a
merge of an `auto-deploy/fishsense-data-processing-workflow-worker-*`
PR (opened by `promote.yml` against
[deploy/compose.data-worker.yml](../../deploy/compose.data-worker.yml))
routes to the `fishsense-data-worker` self-hosted runner, which runs
`docker compose -f compose.data-worker.yml pull && up -d` in
`$DATA_WORKER_DEPLOY_DIR` on the data-worker host.

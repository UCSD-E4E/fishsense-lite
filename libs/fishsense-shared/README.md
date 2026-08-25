# fishsense-shared

Shared helpers used by every FishSense Lite service. Workspace-only — not
published.

## What's in here

| Module | Surface | Notes |
|---|---|---|
| `config` | `IS_DOCKER`, `get_config_path()`, `get_log_path(app_name)`, `path_validator`, `url_condition` | Picks `/e4efs/{config,logs}` when `E4EFS_DOCKER=true`, else cwd / platformdirs. `get_config_path()` falling back to cwd is what causes the repo-root `settings.toml` side-effect — see `CLAUDE.md`. `url_condition` is the lenient http(s)-with-non-empty-hostname validator used by every service's `*.url` setting (see CLAUDE.md for why we don't use `validators.url`). |
| `logging` | `configure_logging(app_name, log_filename=None)`, `configure_log_handler(handler)` | Root logger at DEBUG, UTC timestamps, midnight-rotated file handler + console handler. |
| `temporal` | `build_tls_config(temporal_settings)`, `temporal_namespace(...)`, `ensure_schedule(...)`, `retire_schedule(...)` | `build_tls_config` reads the cert files named by `settings.temporal` and returns a `TLSConfig` (or `None` when `temporal.tls=False`). `ensure_schedule` is idempotent and **refuses to update in place** — an operator must `temporal schedule delete <id>` to change config, so a typo can't silently retire a schedule. `retire_schedule` actively deletes one (used for the dormant slate detector). |
| `exception_group` | `ExceptionGroupErrorLogging(logger)` | Async context manager that flattens `ExceptionGroup`s from `asyncio.TaskGroup` into per-exception log lines. |
| `object_store` | `BaseObjectStoreClient`, `build_s3_client`, `open_client`, the `raw_key` / `slate_pdf_key` / `jpeg_key` helpers | **The Garage key contract between the two workers**, so neither owns it. Each worker subclasses `BaseObjectStoreClient` with only its permitted methods — the api-worker stages and deletes scratch, the data-worker reads scratch and writes JPEGs — and that asymmetry is a real safety boundary. Merged from two drifted copies in 2026-08; one of them leaked a pooled HTTP connection on every `download_slate_pdf` because only the other closed the `StreamingBody`. |
| `preprocess_contracts` | `Preprocess{Laser,Species,Headtail,Slate}ImagesInput`, `ClusterDiveFramesInput`, `Predict*Input`, `LaserPredictionResult` | Workflow-input DTOs for the api-worker → data-worker dispatch. They live here because they are the contract *between* the two workers; per-image DTOs stay in the data-worker because they are internal to its fan-out. |
| `taxonomy` | `is_measurable`, `parse_species_names`, `parse_model_name`, `measurable_species_sql`, `rigid_target_sql`, `MEASURABILITY_CORPUS`, the branch literals | `SpeciesLabel.content_of_image` is read by four things in three languages. `is_measurable` is the **definition of record**; the SQL predicates are approximations of it, and `test_dive_pipeline_status_view.py` runs both over `MEASURABILITY_CORPUS` and asserts they agree. Never spell a marker inline in a controller, view or activity. |
| `ingest_contracts` | `IngestDiveRequest`, `IngestPreflight`, `IngestReport`, `IngestProgress`, `DuplicateOverlap`, `VerifyChecksumsReport`, `VerifyAllDivesReport` | The fishsense-api ↔ api-worker ingest contract: the API starts `IngestDiveWorkflow` and reads its progress, the api-worker runs it, and neither package imports the other. See [docs/ingest.md](../../docs/ingest.md). |

## Conventions enforced

- **`E4EFS_` envvar prefix.** Every service uses `Dynaconf(envvar_prefix="E4EFS", ...)`. The config helpers here assume that.
- **Settings files.** Services load `(get_config_path() / "settings.toml")` and `(get_config_path() / ".secrets.toml")`. Outside Docker that resolves to cwd, so don't run a worker from the repo root unless you want a `settings.toml` to materialize there.
- **Temporal mTLS shape.** `build_tls_config` expects `settings.temporal.{tls, client_cert, client_private_key}`, plus optional `server_root_ca_cert` and `domain`. New workers should reuse the same key names so this helper keeps applying.

## Adding to the public surface

Re-export from `fishsense_shared/__init__.py` and append to `__all__`. Keep
this lib import-cheap — it's loaded eagerly by every service's
`config.py` at module level (Dynaconf validates on first attribute
access; see the CLAUDE.md gotcha).

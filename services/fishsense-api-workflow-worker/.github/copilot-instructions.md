# Copilot Instructions for FishSense API Workflow Worker

## Project Overview

This is a Temporal workflow worker for the FishSense API that manages automated data synchronization workflows. It handles syncing labels from Label Studio projects to the FishSense API, including laser calibration labels and head-tail fish measurement labels.

## Technology Stack

- **Language**: Python 3.13+
- **Workflow Engine**: Temporal (temporalio SDK)
- **Configuration**: Dynaconf (settings.toml for configuration)
- **Package Manager**: uv (pip-compatible)
- **Key Dependencies**:
  - `temporalio`: Temporal workflow and activity framework
  - `fishsense-api-sdk`: FishSense API client (from Git repository)
  - `label-studio-sdk`: Label Studio API client
  - `dynaconf`: Configuration management

## Code Organization

The project follows Temporal's workflow/activity pattern:

- `src/fishsense_api_workflow_worker/`
  - `worker.py`: Main worker entry point, schedules and runs workflows
  - `workflows/`: Temporal workflow definitions (business logic orchestration)
  - `activities/`: Temporal activity definitions (actual work implementations)
  - `models/`: Data models
  - `config.py`: Configuration and logging setup
  - `exception_group_error_logging.py`: Error handling utilities

### Temporal Patterns

- **Workflows** orchestrate activities and handle business logic flow
- **Activities** perform actual work (API calls, data processing)
- Workflows use `@workflow.defn` decorator and `@workflow.run` method
- Activities use `@activity.defn` decorator
- All workflows/activities are registered with the worker in `worker.py`

## Development Practices

### Linting
- Use `pylint` for code quality checks
- Run: `uv run pylint $(git ls-files '*.py')`
- Some pylint rules are disabled inline (e.g., `line-too-long`, `duplicate-code`, `too-few-public-methods`)
- Always respect existing pylint disable comments unless fixing the underlying issue

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Use async/await for all I/O operations
- Use `asyncio.TaskGroup()` for concurrent operations
- Import activities and workflows by their full module path

### Configuration
- Use Dynaconf settings object from `config.py`
- Configuration is loaded from `settings.toml` and environment-specific files
- Secrets use `.secrets.toml` (not committed to Git)
- Access settings via `settings.general.max_workers`, `settings.temporal.host`, etc.

### Error Handling
- Use `ExceptionGroupErrorLogging` context manager for TaskGroup error logging
- Log errors appropriately with `activity.logger` or `workflow.logger`
- Handle API errors gracefully (e.g., `ApiError` from label-studio-sdk)

### Timeouts
- Set appropriate `schedule_to_close_timeout` for activities (typically 10-30 minutes)
- Set `run_timeout` for workflows (typically 30 minutes)

## Key Conventions

1. **Naming**: 
   - Activities end with `_activity` suffix
   - Workflows end with `_workflow` or `Workflow` suffix
   - Use snake_case for functions and variables
   - Use PascalCase for classes

2. **Task Queue**: All workflows and activities use `TASK_QUEUE_NAME = "fishsense_api_queue"`

3. **Scheduling**: 
   - Workflows are scheduled to run periodically (hourly by default)
   - Check if schedules exist before creating new ones

4. **Clients**:
   - FishSense API client: Use `get_fs_client()` context manager
   - Label Studio client: Use `get_ls_client()` (non-async)
   - Temporal client: Created in `worker.py` with TLS configuration

5. **Async Patterns**:
   - Use `await asyncio.to_thread()` for blocking Label Studio SDK calls
   - Use `async with asyncio.TaskGroup()` for concurrent operations
   - Always check `activity.is_cancelled()` in long-running loops

## Testing

- Currently no formal test suite exists
- Manual testing involves running the worker locally with Temporal Dev Server
- Start Temporal Dev Server: `temporal server start-dev`

## Docker & Deployment

- Production Dockerfile: `Dockerfile`
- Development Dockerfile: `Dockerfile.develop`
- Development compose: `docker-compose.develop.yml`
- Worker connects to remote Temporal server with TLS authentication

## Important Notes

- The worker automatically schedules workflows on startup if they don't already exist
- Labels are synced from Label Studio to FishSense API (not bidirectional)
- Each project type (laser vs. head-tail) has separate workflows and activities
- Dashboard config workflow generates configuration files for visualization
- Connection to Temporal requires TLS certificates in production

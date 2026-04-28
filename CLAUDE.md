# fishsense-lite-mono — Claude operating notes

Loose ends from prior sessions that aren't otherwise tracked.

## Repo-root `settings.toml` — do NOT commit

`fishsense_shared.get_config_path()` falls back to `cwd` outside Docker,
so the data-processing-workflow-worker reads `./settings.toml` when run
from the repo root. Running it locally creates this file as a side-effect;
it has prod-y URLs inside.

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

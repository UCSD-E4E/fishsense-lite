#!/usr/bin/env bash
# check.sh — run lint / unit tests / integration tests across the workspace.
#
# Usage:
#   ./check.sh lint         # pylint on Python files changed since origin/main
#                            + ESLint on apps/fishsense-lite-web if any TS/JS changed
#                            (matches .github/workflows/lint.yml + the lint
#                             step in the apps/fishsense-lite-web vitest CI job).
#   ./check.sh unit         # pytest with default markers per Python package
#                            + apps/fishsense-lite-web typecheck + vitest (CI mode).
#   ./check.sh integration  # pytest -m integration (needs the local devcontainer
#                            stack: postgres, temporal, nginx). The fishsense-lite-web
#                            SSR smoke check that runs in integration.yml is a
#                            shell-level curl-and-grep against the local
#                            container; replicate it manually if needed.
#   ./check.sh all          # lint, then unit, then integration (fail-fast across
#                            categories; within a category each package runs
#                            even if a prior one failed, so the report is full)
#
# Notes:
# - Uses `uv run --package <pkg> python -m pytest` not bare `pytest` because
#   the .venv pylint/pytest entrypoints have a stale-shebang failure mode in
#   this devcontainer (see CLAUDE.md history); python -m sidesteps it.
# - Each package's pyproject.toml sets `addopts = "-m 'not integration'"`,
#   so `unit` deselects integration tests automatically.
# - Node steps require Node 22+ on PATH (the devcontainer ships Node 24). If
#   `npm` isn't on PATH the apps/fishsense-lite-web steps are skipped with a notice — they
#   don't fail the run, so a Python-only environment can still ./check.sh.

set -euo pipefail

# (package_name, path) pairs, run in this order.
PACKAGES=(
    "fishsense-api:services/fishsense-api"
    "fishsense-api-workflow-worker:services/fishsense-api-workflow-worker"
    "fishsense-backup-worker:services/fishsense-backup-worker"
    "fishsense-data-processing-workflow-worker:services/fishsense-data-processing-workflow-worker"
    "fishsense-api-sdk:libs/fishsense-api-sdk"
    "fishsense-shared:libs/fishsense-shared"
)

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 2
}

heading() {
    printf '\n=== %s ===\n' "$*"
}

run_lint() {
    heading "lint"
    git fetch --quiet origin main 2>/dev/null || true
    local base
    base="$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD)"
    local rc=0

    # Compare working tree (committed + staged + unstaged) to $base, not
    # HEAD, so a local pre-commit run lints uncommitted edits. CI is
    # unaffected because there the working tree equals HEAD.
    local changed_py
    changed_py="$(git diff --name-only --diff-filter=ACMR "$base" -- '*.py')"
    if [ -z "$changed_py" ]; then
        echo "no Python changes since origin/main; skipping pylint"
    else
        echo "$changed_py" | xargs uv run python -m pylint || rc=$?
    fi

    # ESLint runs full-project (next lint has no incremental mode), but
    # only when something under apps/fishsense-lite-web changed. Skip cleanly when npm
    # isn't installed so a Python-only host can still run check.sh.
    local changed_web
    changed_web="$(git diff --name-only --diff-filter=ACMR "$base" -- 'apps/fishsense-lite-web/*')"
    if [ -z "$changed_web" ]; then
        echo "no apps/fishsense-lite-web changes since origin/main; skipping eslint"
    elif ! command -v npm >/dev/null 2>&1; then
        echo "npm not on PATH; skipping apps/fishsense-lite-web eslint"
    else
        npm --prefix apps/fishsense-lite-web run lint || rc=$?
    fi

    return "$rc"
}

# $1: marker (empty for unit / package-default, "integration" for integration)
run_pytests() {
    local marker="$1"
    local label="${marker:-unit}"
    local failed=0
    local missing=0
    for entry in "${PACKAGES[@]}"; do
        local pkg="${entry%%:*}"
        local path="${entry##*:}"
        if [ ! -d "$path/tests" ]; then
            missing=$((missing + 1))
            continue
        fi
        heading "$label tests: $pkg"
        # Pytest exit 5 = "no tests collected." For marker-filtered runs that
        # just means the package has no tests tagged with this marker (e.g.
        # fishsense-shared has unit tests but no integration tests). Don't
        # count exit 5 as failure — only real test failures (1) and harness
        # errors (2,3,4) should fail the package.
        local rc=0
        if [ -n "$marker" ]; then
            uv run --package "$pkg" python -m pytest "$path/tests/" -m "$marker" \
                || rc=$?
            [ "$rc" -eq 5 ] && rc=0
        else
            uv run --package "$pkg" python -m pytest "$path/tests/" \
                || rc=$?
        fi
        [ "$rc" -ne 0 ] && failed=$((failed + 1))
    done
    if [ "$missing" -gt 0 ]; then
        echo "(skipped $missing package(s) with no tests/ dir)"
    fi
    if [ "$failed" -gt 0 ]; then
        echo "FAILED: $failed package(s)"
        return 1
    fi
}

run_npm_unit() {
    heading "unit tests: apps/fishsense-lite-web (typecheck + vitest)"
    if ! command -v npm >/dev/null 2>&1; then
        echo "npm not on PATH; skipping apps/fishsense-lite-web unit tests"
        return 0
    fi
    local rc=0
    npm --prefix apps/fishsense-lite-web run typecheck || rc=$?
    npm --prefix apps/fishsense-lite-web test || rc=$?
    return "$rc"
}

# Wrap run_pytests + run_npm_unit so a Python failure doesn't suppress
# the JS run (and vice versa) — the goal is a complete report per
# invocation, mirroring the per-package behavior already inside
# run_pytests.
run_unit() {
    local rc=0
    run_pytests "" || rc=$?
    run_npm_unit || rc=$?
    return "$rc"
}

case "${1:-}" in
    lint)         run_lint ;;
    unit)         run_unit ;;
    integration)  run_pytests "integration" ;;
    all)          run_lint && run_unit && run_pytests "integration" ;;
    -h|--help|"") usage ;;
    *)            echo "unknown command: $1" >&2; usage ;;
esac

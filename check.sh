#!/usr/bin/env bash
# check.sh — run lint / unit tests / integration tests across the workspace.
#
# Usage:
#   ./check.sh lint         # pylint on Python files changed since origin/main
#                            (matches .github/workflows/lint.yml)
#   ./check.sh unit         # pytest with default markers per package (CI mode)
#   ./check.sh integration  # pytest -m integration (needs the local devcontainer
#                            stack: postgres, temporal, nginx)
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
    local changed
    # Compare working tree (committed + staged + unstaged) to $base, not
    # HEAD, so a local pre-commit run lints uncommitted edits. CI is
    # unaffected because there the working tree equals HEAD.
    changed="$(git diff --name-only --diff-filter=ACMR "$base" -- '*.py')"
    if [ -z "$changed" ]; then
        echo "no Python changes since origin/main; skipping pylint"
        return 0
    fi
    echo "$changed" | xargs uv run python -m pylint
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

case "${1:-}" in
    lint)         run_lint ;;
    unit)         run_pytests "" ;;
    integration)  run_pytests "integration" ;;
    all)          run_lint && run_pytests "" && run_pytests "integration" ;;
    -h|--help|"") usage ;;
    *)            echo "unknown command: $1" >&2; usage ;;
esac

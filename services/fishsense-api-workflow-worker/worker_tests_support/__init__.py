"""Test helpers shared across this package's test modules.

Named `worker_tests_support`, not `tests_support`, and deliberately so. The
changed-files pylint run passes every package's files in one invocation, and
two packages sharing a module name means whichever lands on sys.path first
wins — a sibling import then resolves against the wrong package and fails.
`services/fishsense-api/tests_support` already holds that name, so this one is
distinct. Same reasoning that kept these out of `tests/` in the first place.
"""

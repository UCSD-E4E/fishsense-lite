"""Test helpers shared across this package's test modules.

Deliberately NOT inside `tests/`. Six packages in this workspace each have a
`tests` package, and the changed-files pylint run passes them all in one
invocation — whichever `tests/` lands on sys.path first wins, so a sibling
import from `tests.<module>` resolves against another package's tests and
fails to import. A workspace-unique package name has nothing to collide with.
"""

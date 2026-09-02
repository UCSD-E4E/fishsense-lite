"""Every activity a workflow names by string must be registered on the worker.

Workflows dispatch activities by *string*, so a missing registration is
invisible everywhere a test normally looks: the module imports fine, the
workflow contract tests stub the activity by the same string and pass, lint is
clean, and CI is green. It surfaces only when the workflow runs in production
and Temporal answers `NotFoundError: Activity function ... is not registered on
this worker`.

That is not hypothetical. `backfill_laser_predictions_for_dive_activity`
shipped registered as a *workflow* but never as an *activity*, and the failure
mode was quiet and expensive: the stage-0.1 predict parent re-predicted all 259
images of prod dive 442, persisted them, then died on the unregistered
backfill. So the database was updated, the labelers' pre-annotations were not
(which was the entire point of that step), `cleanup_raw` never ran and left
1,094 raw `.ORF` objects staged in Garage, and the dive then drained out of the
cohort — leaving nothing to retry it.

This test reads the activity names out of the workflow sources and checks them
against what `worker.py` actually registers.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "fishsense_api_workflow_worker"
_WORKFLOWS = _SRC / "workflows"


def _string_activity_names() -> dict[str, set[str]]:
    """Activity-name literals passed to `execute_activity` / `start_activity`,
    keyed by the workflow module that names them.

    String literals only. A name passed as a variable cannot be resolved
    statically, and quietly skipping those is better than guessing wrong --
    every dispatch in this package is a literal today.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(_WORKFLOWS.glob("*.py")):
        tree = ast.parse(path.read_text())
        names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            attr = func.attr if isinstance(func, ast.Attribute) else None
            if attr not in {"execute_activity", "start_activity"}:
                continue
            if node.args and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str):
                    names.add(value)
        if names:
            found[path.name] = names
    return found


def _registered_activity_names() -> set[str]:
    """Names Temporal will actually serve, taken from the real registration."""
    from fishsense_api_workflow_worker import worker as worker_module

    source = (_SRC / "worker.py").read_text()
    tree = ast.parse(source)
    registered: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != "activities":
            continue
        if isinstance(node.value, ast.List):
            for element in node.value.elts:
                if isinstance(element, ast.Name):
                    registered.add(element.id)
    assert registered, "no activities= list found in worker.py"
    # The decorated name is what Temporal registers, and in this package the
    # symbol and the activity name are the same; assert that rather than
    # assume it.
    for name in sorted(registered):
        assert hasattr(worker_module, name), f"{name} registered but not imported"
    return registered


def test_worker_registers_some_activities():
    """Guard the guard: if the parse silently found nothing, every assertion
    below would pass vacuously."""
    assert len(_registered_activity_names()) > 30


def test_workflows_reference_some_activities():
    named = _string_activity_names()
    assert named, "parsed no activity names out of the workflows package"


@pytest.mark.parametrize(
    "module,name",
    sorted(
        (module, name)
        for module, names in _string_activity_names().items()
        for name in names
    ),
)
def test_every_named_activity_is_registered(module, name):
    registered = _registered_activity_names()
    assert name in registered, (
        f"{module} dispatches '{name}', which worker.py does not register. "
        "Temporal resolves activities by string, so this fails only at "
        "runtime, in production, after the workflow has already done its work."
    )

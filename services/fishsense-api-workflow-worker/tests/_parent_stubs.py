"""Shared stubs for the cross-worker parent-workflow contract tests.

Every one of these parents runs the same play — selector activity, wake the
data-worker, dispatch a child on the worker's task queue — so their contract
tests were near-identical scaffolding with one activity name swapped. That is
textual duplication `duplicate-code` cannot normally see, because the differing
strings hide it; it surfaced the moment the light-queue split moved two of the
copies in one PR.

Extracted rather than exempted. The three helpers below are the whole clone,
and the only thing that actually varied between copies was the selector's
activity name — now a parameter.

Note the stubs are *named* Temporal activities, not plain functions: the
parents call them by string across the workflow sandbox, so the registered name
is the contract and the Python identifier is irrelevant.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from temporalio import activity


def make_recording_activity(captures: List[tuple]) -> Callable:
    """`_record_child_dispatch`: how a stub child reports what it was handed.

    The child runs inside the workflow sandbox, so it cannot append to a list
    in the test process directly. Forwarding through an activity is what lets
    the contract test observe the dispatched workflow id and payload.
    """

    @activity.defn(name="_record_child_dispatch")
    async def record_child_dispatch(workflow_id: str, dive_id: int) -> None:
        captures.append((workflow_id, dive_id))

    return record_child_dispatch


@activity.defn(name="ensure_light_worker_running_activity")
async def stub_wake_light_worker() -> int:
    """Stand-in for the light-queue scale-up. Returns 0, the same as the real
    activity does when k8s scaling isn't configured."""
    return 0


@activity.defn(name="ensure_data_worker_running_activity")
async def stub_wake_data_worker() -> int:
    """Stand-in for the per-image scale-up, for parents still on that queue."""
    return 0


def make_stub_selector(
    activity_name: str, selector_result: Optional[int]
) -> Tuple[Callable, List[None]]:
    """A cohort selector that returns `selector_result` and counts its calls.

    `activity_name` was the only real difference between the copies of this
    helper — each stage's selector is registered under its own name.
    """
    selector_calls: List[None] = []

    @activity.defn(name=activity_name)
    async def stub_select() -> Optional[int]:
        selector_calls.append(None)
        return selector_result

    return stub_select, selector_calls

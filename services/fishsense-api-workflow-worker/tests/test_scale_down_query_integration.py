"""Integration test for the scale-down activity's Temporal visibility query.

`scale_down_data_worker_if_idle_activity` decides "is the data-worker
busy?" by running a list-filter against Temporal —
`TaskQueue = "fishsense_data_processing_queue" and (ExecutionStatus =
"Running" or CloseTime > "<now - cooldown>")`. Whether that string
parses and whether the visibility store actually answers it correctly
is the kind of thing you only find out against a real server, so this
exercises it against the devcontainer's Temporal (closer to prod's
config than the in-memory time-skipping test server).

`@pytest.mark.integration` — needs the local stack (`temporal:7233`).
Visibility is eventually consistent, so the assertions poll.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from fishsense_api_workflow_worker.activities.scale_down_data_worker_if_idle_activity import (  # noqa: E501  pylint: disable=line-too-long
    DATA_WORKER_TASK_QUEUE,
    _data_worker_task_queue_busy,
)
from fishsense_api_workflow_worker.config import settings
from fishsense_shared import build_tls_config
from temporalio.client import Client

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _temporal_target_from_ci_env(monkeypatch):
    """Point `settings.temporal` at the right host for this environment.

    `_data_worker_task_queue_busy` (and `_connect` below) connect via
    `settings.temporal.host` — which conftest hard-sets to `temporal`,
    resolvable only inside the docker network. The integration CI job
    (`integration.yml`) runs pytest on the runner host and reaches the
    compose services through published localhost ports, exporting
    `FISHSENSE_TEMPORAL_HOST` / `FISHSENSE_TEMPORAL_PORT` for exactly
    this — honor those (the rest of the integration suite does too).
    Inside the devcontainer they're unset, so this is a no-op and the
    conftest's `temporal` hostname stands.
    """
    host = os.environ.get("FISHSENSE_TEMPORAL_HOST")
    port = os.environ.get("FISHSENSE_TEMPORAL_PORT")
    if not host and not port:
        return
    if host:
        monkeypatch.setenv("E4EFS_TEMPORAL__HOST", host)
    if port:
        monkeypatch.setenv("E4EFS_TEMPORAL__PORT", port)
    # pylint: disable=import-outside-toplevel
    from fishsense_api_workflow_worker import config as cfg

    cfg.settings.reload()

# Big enough that any workflow closed during this test counts as
# "recently closed", small enough to stay an int Temporal accepts.
_LONG_COOLDOWN_MIN = 10 * 365 * 24 * 60


async def _connect() -> Client:
    return await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}",
        tls=build_tls_config(settings.temporal),
    )


async def _poll(predicate, *, timeout_s: float = 20.0, interval_s: float = 0.5) -> bool:
    """Poll an async predicate until True or timeout."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if await predicate():
            return True
        await asyncio.sleep(interval_s)
    return False


async def _terminate_running_on_data_queue(client: Client) -> None:
    """Clean up any Running workflows on the data-worker queue (e.g. left
    by a crashed prior run of this test) so we start from a known state."""
    async for wf in client.list_workflows(
        query=f'TaskQueue = "{DATA_WORKER_TASK_QUEUE}" and ExecutionStatus = "Running"'
    ):
        try:
            await client.get_workflow_handle(
                wf.id, run_id=wf.run_id
            ).terminate("test cleanup")
        except Exception:  # pylint: disable=broad-except
            pass


async def _busy(cooldown: int = 0) -> bool:
    return await _data_worker_task_queue_busy(cooldown)


async def _not_busy(cooldown: int = 0) -> bool:
    return not await _data_worker_task_queue_busy(cooldown)


async def test_busy_query_tracks_running_and_recently_closed_on_data_queue():
    client = await _connect()
    await _terminate_running_on_data_queue(client)
    # Quiet baseline (give visibility a moment to reflect the cleanup).
    assert await _poll(_not_busy), "expected the data-worker queue to be quiet to start"

    wf_id = f"k8s-scaling-it-{uuid.uuid4()}"
    # The type need not be registered anywhere — `start_workflow` just
    # records the execution; with no worker it sits Running until we
    # terminate it.
    handle = await client.start_workflow(
        "ScaleDownProbeWorkflow",
        id=wf_id,
        task_queue=DATA_WORKER_TASK_QUEUE,
    )
    try:
        assert await _poll(_busy), (
            "the query should see the Running workflow on "
            f"{DATA_WORKER_TASK_QUEUE}"
        )
    finally:
        await handle.terminate("end of test")

    # cooldown 0: a workflow that closed in the past is NOT "recent",
    # so the queue reads quiet again.
    assert await _poll(_not_busy), (
        "after terminate, cooldown=0 should report the queue quiet"
    )
    # huge cooldown: the just-terminated workflow IS within the window,
    # so the CloseTime clause fires and the queue reads busy.
    assert await _poll(lambda: _busy(_LONG_COOLDOWN_MIN)), (
        "a recently-closed workflow within the cooldown should count as busy"
    )

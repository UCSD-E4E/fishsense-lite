"""Data-worker Temporal worker construction and run loop.

Single ``build_worker`` construction point (shared with tests) plus the
``main`` scale-to-zero-friendly run loop. Caps activity concurrency to keep
the CPU-heavy per-image rectify/decode work under the pod's memory limit.

The worker runs in one of two roles — ``cpu`` or ``gpu`` — each polling its
own task queue, or ``all`` (both, in one process) for local development. What
each role registers lives in `roles.py`; this module only turns a role into
Temporal ``Worker`` objects and runs them.
"""

import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from datetime import timedelta

from fishsense_shared import (
    DATA_PROCESSING_TASK_QUEUE,
    build_tls_config,
    temporal_namespace,
)
from temporalio.client import Client
from temporalio.worker import Worker

from fishsense_data_processing_workflow_worker import roles
from fishsense_data_processing_workflow_worker.config import configure_logging, settings

# Retained for the CPU queue's historical name so existing callers and tests
# keep reading one constant. `roles.queue_for_role` is the general form.
TASK_QUEUE_NAME = DATA_PROCESSING_TASK_QUEUE

# How long in-flight activities get to finish when the worker is asked to
# stop. On NRP the api-worker scales this deployment to zero when idle, so
# a scale-down delivers SIGTERM mid-activity; without a graceful window the
# rectify/measure work in progress is cancelled immediately and re-queued
# (idempotent, so safe — this just avoids throwing away a nearly-done image).
# The k8s Deployment's terminationGracePeriodSeconds must be >= this.
GRACEFUL_SHUTDOWN_TIMEOUT = timedelta(seconds=30)

# Cap on activities executed at once. The per-image activities are async and
# offload full-res rawpy decode + opencv rectify via `asyncio.to_thread`, each
# peaking at ~1-3 GB. The Temporal SDK default (100) let a burst run ~8-9
# decodes concurrently, which blew the pod's memory limit → OOMKilled →
# CrashLoopBackOff (the whole worker died on startup within seconds, so nothing
# drained). A cap of 4 was still too high: on 2026-07-21 headtail preprocessing
# OOMKilled (exit 137) ~22s after start and CrashLoopBackOff'd 17 times, which
# starved `fishsense_data_processing_queue` and timed out the
# ValidateLaserLabelsForDive children after 17 min. It cannot self-heal —
# Temporal redelivers the same heavy tasks on restart. Keep this low so peak
# memory stays bounded and scale throughput horizontally instead
# (`kubernetes.active_replicas` on the api-worker); tune via
# `general.max_concurrent_activities`.
DEFAULT_MAX_CONCURRENT_ACTIVITIES = 2


async def schedule_workflows(_: Client):
    """Schedule workflows for the worker.

    The data-worker owns no recurring schedules — those live on the
    always-up api-worker — so scaling this service to zero has no effect
    on schedule registration. Kept as a hook for symmetry with the other
    workers.
    """


def _build(
    client: Client,
    activity_executor: ThreadPoolExecutor,
    max_concurrent_activities: int,
    registration: roles.Registration,
) -> Worker:
    """Turn one role's wiring into a Temporal ``Worker``."""
    return Worker(
        client,
        task_queue=registration.task_queue,
        max_concurrent_activities=max_concurrent_activities,
        workflows=list(registration.workflows),
        activity_executor=activity_executor,
        activities=list(registration.activities),
        graceful_shutdown_timeout=GRACEFUL_SHUTDOWN_TIMEOUT,
    )


def build_worker(
    client: Client,
    activity_executor: ThreadPoolExecutor,
    max_concurrent_activities: int = DEFAULT_MAX_CONCURRENT_ACTIVITIES,
    role: str = roles.ROLE_CPU,
) -> Worker:
    """Construct one role's Temporal worker.

    Single construction point so the worker config (workflows, activities,
    graceful-shutdown window, activity concurrency cap) is exercised by tests
    without standing up the full ``main`` loop.

    Raises ``ValueError`` for ``all`` — that is two queues, so it has no single
    answer. Use ``build_workers``.
    """
    return _build(
        client,
        activity_executor,
        max_concurrent_activities,
        roles.registration_for_role(role),
    )


def build_workers(
    client: Client,
    activity_executor: ThreadPoolExecutor,
    max_concurrent_activities: int = DEFAULT_MAX_CONCURRENT_ACTIVITIES,
    role: str = roles.ROLE_ALL,
) -> list[Worker]:
    """Every worker this process should run for ``role``.

    One entry for ``cpu``/``gpu``; both for ``all``, which is what the
    devcontainer and the integration tests use so a local process still serves
    every queue the api-worker dispatches to.
    """
    return [
        _build(client, activity_executor, max_concurrent_activities, registration)
        for registration in roles.registrations_for_role(role)
    ]


async def main():
    """Main entry point for the worker."""

    configure_logging()
    log = logging.getLogger()

    role = settings.general.get("role", roles.ROLE_ALL)
    tls_config = build_tls_config(settings.temporal)

    log.info(
        "connecting to Temporal host=%s:%d tls=%s role=%s",
        settings.temporal.host,
        settings.temporal.port,
        bool(tls_config),
        role,
    )
    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}",
        tls=tls_config,
        namespace=temporal_namespace(settings.temporal),
    )

    interrupt_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, interrupt_event.set)

    with ThreadPoolExecutor(max_workers=settings.general.max_workers) as executor:
        workers = build_workers(
            client,
            executor,
            settings.general.get(
                "max_concurrent_activities", DEFAULT_MAX_CONCURRENT_ACTIVITIES
            ),
            role=role,
        )
        # AsyncExitStack rather than a single `async with`: the `all` role runs
        # two workers (one per queue) in this one process, and both must be
        # entered and — importantly — drained on the way out. Exiting the stack
        # gives each its own graceful-shutdown window.
        async with AsyncExitStack() as stack:
            for worker in workers:
                await stack.enter_async_context(worker)
            log.info(
                "Worker started on %s, scheduling workflows...",
                ", ".join(worker.config()["task_queue"] for worker in workers),
            )
            await schedule_workflows(client)
            await interrupt_event.wait()
            log.info(
                "shutdown signal received; draining (graceful_shutdown_timeout=%s)",
                GRACEFUL_SHUTDOWN_TIMEOUT,
            )


def run():
    """Run the worker."""
    asyncio.run(main())

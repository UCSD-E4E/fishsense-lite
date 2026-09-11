"""Activity to delete a dive's staged raw `.ORF` bytes from the Garage
object store (the scratch copy only).

Runs on the api-worker after the data-worker has produced the processed
JPEGs — frees object-store space by dropping the reproducible-from-NAS
raw inputs. The processed JPEGs intentionally stay in Garage (LS reads
them via presign); their retention is a separate operational decision
(see `MEMORY.md` / `jpeg-retention-open-question`).

NAS safety invariant: this only deletes the Garage `raw/{checksum}.ORF`
*scratch* objects. The NAS source `.ORF` is never touched — there is no
NAS-delete path anywhere on the api-worker.

S3 delete_object is idempotent (deleting an absent key is a success),
so this is naturally safe under retries.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from temporalio import activity

from temporalio.client import Client

from fishsense_shared import build_tls_config, temporal_namespace

from fishsense_api_workflow_worker.activities.utils import get_fs_client
from fishsense_api_workflow_worker.config import settings
from fishsense_api_workflow_worker.object_store import open_object_store_client

CLEANUP_CONCURRENCY = 8

__all__ = [
    "CLEANUP_CONCURRENCY",
    "raw_scratch_reader_ids",
    "CleanupRawBytesResult",
    "build_scratch_in_use_query",
    "scratch_in_use",
    "cleanup_raw_bytes_for_dive_activity",
]


def raw_scratch_reader_ids(dive_id: int) -> list[str]:
    """Deterministic ids of every child that reads this dive's raw scratch.

    Preprocess, predict and both checkerboard children `download_raw(checksum)`
    from `raw/{checksum}.ORF`. Scratch is keyed per **dive**, not per stage --
    which is what lets a dive in several cohorts stage once and report
    `skipped_already_present` afterwards -- so deleting it is a cross-stage
    act.

    **Every new child that reads raw scratch must be added here.** Omitting one
    does not fail loudly: the omitted child is simply invisible to every other
    stage's cleanup, which then deletes the `.ORF`s out from under it mid-read
    and kills the render with `NoSuchKey`, costing the whole dive's NAS staging
    to redo. That is the dive-442 incident (2026-09-07) this list exists to
    prevent, and the checkerboard pair sat in exactly that gap -- the
    calibration child from 2026-09-07, the lattice child from 2026-09-11 --
    until both were added on 2026-09-11.
    """
    return [
        f"preprocess-laser-{dive_id}",
        f"preprocess-species-{dive_id}",
        f"preprocess-headtail-{dive_id}",
        f"preprocess-slate-{dive_id}",
        f"predict-laser-{dive_id}",
        f"predict-slate-{dive_id}",
        f"perform-checkerboard-calibration-{dive_id}",
        f"verify-checkerboard-lattice-{dive_id}",
    ]


def build_scratch_in_use_query(dive_id: int) -> str:
    """Temporal visibility query: is any sibling child still reading?

    `WorkflowId IN (...)`, deliberately not a prefix match -- dive 44 must not
    hold dive 442's scratch open, nor 4420's.
    """
    ids = ", ".join(f"'{wid}'" for wid in raw_scratch_reader_ids(dive_id))
    return f'ExecutionStatus = "Running" and WorkflowId in ({ids})'


async def scratch_in_use(dive_id: int) -> str | None:
    """The id of a still-running sibling child, or None if the scratch is free.

    The child-id sentinel in `_dispatch.dispatch_child` cannot cover this: the
    ids differ across stages, so `WorkflowAlreadyStartedError` is never raised.
    Gated here rather than at each call site because it is one place, it covers
    every caller including future ones, and it adds no workflow command -- so
    no in-flight parent's replay contract changes.

    **Fails closed.** If Temporal cannot be reached the answer is unknown, and
    the two ways of being wrong are not symmetric: deleting scratch a live
    child is reading kills a render silently and costs the whole dive's NAS
    staging to redo, while keeping it costs object-store space until the next
    firing re-stages -- which is cheap, because staging reports
    `skipped_already_present` and re-uses what is there. So an unreachable
    Temporal blocks the delete rather than waving it through.
    """
    try:
        client = await Client.connect(
            f"{settings.temporal.host}:{settings.temporal.port}",
            tls=build_tls_config(settings.temporal),
            namespace=temporal_namespace(settings.temporal),
        )
        async for w in client.list_workflows(query=build_scratch_in_use_query(dive_id)):
            return w.id
    except Exception as exc:  # pylint: disable=broad-except
        activity.logger.warning(
            "cannot determine whether dive_id=%d scratch is in use (%s); "
            "declining to delete",
            dive_id,
            type(exc).__name__,
        )
        return "<temporal-unreachable>"
    return None


@dataclass
class CleanupRawBytesResult:
    """Per-dive cleanup summary."""

    deleted: int  # scratch raw objects deleted from Garage


@activity.defn
async def cleanup_raw_bytes_for_dive_activity(
    dive_id: int,
) -> CleanupRawBytesResult:
    holder = await scratch_in_use(dive_id)
    if holder is not None:
        # Another stage's child is still reading these objects. Deleting now is
        # what killed prod dive 442's render (NoSuchKey, 2026-09-07). Whichever
        # stage finishes last does the cleanup; if that one fails instead the
        # scratch leaks until the next firing re-stages, which is cheap.
        activity.logger.info(
            "skipping raw cleanup dive_id=%d: %s is still reading the scratch",
            dive_id,
            holder,
        )
        return CleanupRawBytesResult(deleted=0)

    async with get_fs_client() as fs:
        images = await fs.images.get(dive_id=dive_id) or []
        # Deliberately NOT filtered to canonical images, unlike staging and
        # the resolvers. Cleanup should be broader than whatever produced the
        # scratch, so it still evicts objects staged before the canonical gate
        # existed. Harmless either way -- scratch is keyed by checksum, and a
        # duplicate shares its canonical twin's key -- but broader is the safe
        # direction for a delete that only touches Garage scratch.
    activity.logger.info(
        "cleaning up raw bytes dive_id=%d images=%d", dive_id, len(images)
    )

    sem = asyncio.Semaphore(CLEANUP_CONCURRENCY)
    deleted = 0

    exchange = open_object_store_client()

    async def _delete_one(image) -> None:
        nonlocal deleted
        if not image.checksum:
            activity.heartbeat()
            return

        async with sem:
            ok = await exchange.delete_raw(image.checksum)
            if ok:
                deleted += 1
            activity.heartbeat()

    async with asyncio.TaskGroup() as tg:
        for image in images:
            tg.create_task(_delete_one(image))

    activity.logger.info("raw cleanup done dive_id=%d deleted=%d", dive_id, deleted)
    return CleanupRawBytesResult(deleted=deleted)

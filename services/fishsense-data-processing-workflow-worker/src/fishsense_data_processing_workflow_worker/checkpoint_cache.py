"""Fetch a model checkpoint from the object store, cached on a volume.

Large weights are kept in Garage rather than baked into the worker image, for
two reasons that both matter here. This Deployment scales to zero every hour,
so a multi-gigabyte image layer is re-pulled on every cold start, and the GPU
start timeout exists partly so an image pull is not mistaken for an outage.
And weights whose upstream distribution is gated should not travel inside a
pullable artifact at all.

The cost of that choice is a download on first use, which a PVC turns into a
download *once per volume* rather than once per pod. This module is the bit in
between: ask for a checkpoint, get a local path, download only if it is not
already there.

Three properties the tests pin, each of which is a way a naive version breaks
on a volume that is shared and long-lived:

* the **version is in the cache path**, so bumped weights can never be served
  from a stale file;
* the download lands via a temporary file and an atomic rename, so a failed or
  interrupted transfer leaves nothing behind rather than a truncated file that
  later reads as a cache hit;
* concurrent callers download **once** — activities run in a real
  ThreadPoolExecutor, so a cold pod arrives here with its whole first batch at
  the same moment.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

# One lock per (name, version, filename). Module-level because the point is to
# serialise callers within a process; the atomic rename is what makes two
# *processes* sharing a ReadWriteMany volume safe.
_LOCKS: dict[tuple[str, str, str], asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


async def _lock_for(key: tuple[str, str, str]) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, asyncio.Lock())


def checkpoint_path(
    cache_dir: Path | str, name: str, version: str, filename: str
) -> Path:
    """Where a given checkpoint is cached. Pure; makes no filesystem calls."""
    return Path(cache_dir) / name / version / filename


async def ensure_checkpoint(
    store, cache_dir: Path | str, name: str, version: str, filename: str
) -> Path:
    """Return a local path to the checkpoint, downloading it if needed.

    `store` is anything with an async `download_model(name, version, filename)`
    — the data-worker's `ObjectStoreClient` in production, a stub in tests.
    """
    target = checkpoint_path(cache_dir, name, version, filename)
    if target.exists():
        return target

    async with await _lock_for((name, version, filename)):
        # Re-check inside the lock: the caller that lost the race has nothing
        # left to do.
        if target.exists():
            return target

        target.parent.mkdir(parents=True, exist_ok=True)
        _log.info("downloading checkpoint %s/%s/%s", name, version, filename)
        data = await store.download_model(name, version, filename)

        # Same directory as the target, so the rename cannot cross filesystems.
        tmp = target.with_name(f".{target.name}.{os.getpid()}.partial")

        def _write() -> None:
            try:
                tmp.write_bytes(data)
                os.replace(tmp, target)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise

        # Off the loop: this is gigabytes. Writing it inline blocks every other
        # coroutine in the worker for the duration, including the Temporal
        # heartbeats that keep the activity from being declared lost.
        await asyncio.to_thread(_write)

        _log.info("cached checkpoint at %s (%d bytes)", target, len(data))
        return target

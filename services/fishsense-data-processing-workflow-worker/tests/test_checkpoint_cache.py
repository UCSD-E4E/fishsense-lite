"""Tests for the on-volume model checkpoint cache.

Weights live in Garage, not in the image (see `object_store.model_key`), and
are cached on a PVC so a Deployment that scales to zero does not re-download
gigabytes on every cold start.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from fishsense_data_processing_workflow_worker.checkpoint_cache import (
    ensure_checkpoint,
)


class _FakeStore:
    def __init__(self, payload: bytes = b"weights"):
        self.payload = payload
        self.calls: list[tuple[str, str, str]] = []

    async def download_model(self, name: str, version: str, filename: str) -> bytes:
        self.calls.append((name, version, filename))
        return self.payload


def test_downloads_when_absent(tmp_path: Path):
    store = _FakeStore()
    path = asyncio.run(
        ensure_checkpoint(store, tmp_path, "sam3", "v1", "sam3.pt")
    )
    assert path.read_bytes() == b"weights"
    assert store.calls == [("sam3", "v1", "sam3.pt")]


def test_second_call_uses_the_cache(tmp_path: Path):
    """The whole point of the volume — a cold start must not re-download."""
    store = _FakeStore()
    asyncio.run(ensure_checkpoint(store, tmp_path, "sam3", "v1", "sam3.pt"))
    asyncio.run(ensure_checkpoint(store, tmp_path, "sam3", "v1", "sam3.pt"))
    assert len(store.calls) == 1


def test_version_is_part_of_the_cache_path(tmp_path: Path):
    """New weights are a new object *and* a new cached file.

    If the version were not in the path, a bumped checkpoint would silently
    read the old bytes off the volume forever — the failure a cache keyed only
    on filename invites.
    """
    store = _FakeStore(b"v1-bytes")
    p1 = asyncio.run(ensure_checkpoint(store, tmp_path, "sam3", "v1", "sam3.pt"))
    store.payload = b"v2-bytes"
    p2 = asyncio.run(ensure_checkpoint(store, tmp_path, "sam3", "v2", "sam3.pt"))

    assert p1 != p2
    assert p1.read_bytes() == b"v1-bytes"
    assert p2.read_bytes() == b"v2-bytes"
    assert len(store.calls) == 2


def test_partial_download_is_not_left_behind(tmp_path: Path):
    """A failed download must not leave a truncated file that later reads as
    a valid cache hit — the volume is shared and long-lived, so a half-written
    checkpoint would poison every subsequent pod.
    """

    class _Failing(_FakeStore):
        async def download_model(self, name, version, filename):
            raise RuntimeError("connection reset")

    with pytest.raises(RuntimeError):
        asyncio.run(ensure_checkpoint(_Failing(), tmp_path, "sam3", "v1", "sam3.pt"))

    assert list(tmp_path.rglob("*.pt")) == []


def test_concurrent_callers_download_once(tmp_path: Path):
    """Activities run in a real ThreadPoolExecutor, so a cold pod enters this
    with the whole first batch at once. Unguarded, each would download its own
    copy of a multi-gigabyte file."""
    store = _FakeStore()

    async def _race():
        return await asyncio.gather(
            *(
                ensure_checkpoint(store, tmp_path, "sam3", "v1", "sam3.pt")
                for _ in range(8)
            )
        )

    paths = asyncio.run(_race())
    assert len({str(p) for p in paths}) == 1
    assert len(store.calls) == 1


def test_version_history_names_the_pinned_checkpoint():
    """The predictor version must record which checkpoint it was pinned to.

    `HEADTAIL_PREDICTOR_VERSION` is the only staleness gate — the cohort keys
    on a mismatch, and `headtail_model_version_tag()` is the Label Studio
    idempotency key. Swapping the checkpoint without bumping it therefore
    leaves rows produced by the *old* weights reading as current forever, and
    their seeded tasks keeping the old keypoints.

    Version 1 recorded only "SAM3", which is ambiguous: upstream publishes two
    non-interchangeable checkpoints (`sam3.pt`, `sam3.1_multiplex.pt`) and the
    loader accepts either with `strict=False`. This asserts the history names
    the file the deployment actually pins, so that ambiguity cannot recur.
    """
    from fishsense_shared import headtail_predictor

    from fishsense_data_processing_workflow_worker.config import _VALIDATORS

    defaults = {v.names[0]: v.default for v in _VALIDATORS}
    filename = defaults["sam3.checkpoint_filename"]

    history = headtail_predictor.__doc__ or ""
    source = Path(headtail_predictor.__file__).read_text(encoding="utf-8")
    assert filename in history or filename in source, (
        f"checkpoint {filename!r} is not named in headtail_predictor.py — "
        "bump HEADTAIL_PREDICTOR_VERSION and record the checkpoint in its "
        "history when the pinned weights change"
    )

"""GPU -> CPU fallback policy for the data-worker's predict queue.

`fishsense_data_processing_gpu_queue` is served by **two** Deployments running
the same image in the same ``gpu`` role: one that requests
``nvidia.com/gpu: 1`` and one that requests none and runs the same torch
checkpoint on the CPU. Exactly one of them is scaled up at a time. This module
decides which.

It exists because the GPU one can fail to start for reasons entirely outside
our control — NRP has no free Turing-or-newer card, our quota is exhausted, the
node pool is drained, the image tag is bad — and when it does, the queue never
drains. `fishsense_core.laser.LaserDetector` already picks
``"cuda" if torch.cuda.is_available() else "cpu"`` and `predict_laser_image`
passes no device, so the CPU Deployment produces the *same predictions*, just
slowly. Slow predictions beat none: the requirement this implements is that
nothing in the pipeline is ever blocked on a GPU being available.

**The state lives in annotations on the GPU Deployment**, not in this process:
a counter that resets whenever the api-worker restarts or the slot converges
would never accumulate across a multi-hour outage, which is exactly the case
the fallback is for. Annotations also make the state operator-visible
(`kubectl describe`) and operator-editable — an operator can force a fallback
with `kubectl annotate ... gpu-start-failures=3 --overwrite`, or end one early
by removing `gpu-fallback-until`.

`decide` is pure, and deliberately so: the interesting behavior is a sequence
spanning hours (wedge, count, flip, hold, expire, probe) that cannot be
reproduced quickly against a real cluster. `tests/test_gpu_fallback.py` walks
the whole state machine in milliseconds.

Two asymmetries worth keeping:

* Counting is per *observation*, not per second of wedge — the wedge clock
  restarts each time a failure is counted, so one continuous outage counts once
  per parent firing (hourly) rather than instantly exhausting the budget.
* The fallback window **expires**. Without it, one bad afternoon on NRP would
  strand the pipeline on CPU inference permanently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Final, Mapping

_log = logging.getLogger(__name__)

# Annotation keys. Prefixed with our domain the same way
# `deploy/incus/nrp_cert_sync` stamps `fishsense.e4e.ucsd.edu/leaf-sha256`.
ANNOTATION_PREFIX: Final = "fishsense.e4e.ucsd.edu/"
FAILURES_ANNOTATION: Final = f"{ANNOTATION_PREFIX}gpu-start-failures"
WEDGED_SINCE_ANNOTATION: Final = f"{ANNOTATION_PREFIX}gpu-wedged-since"
FALLBACK_UNTIL_ANNOTATION: Final = f"{ANNOTATION_PREFIX}gpu-fallback-until"

MODE_GPU: Final = "gpu"
MODE_CPU_FALLBACK: Final = "cpu_fallback"
#: Neither Deployment came up inside the start timeout. The caller should skip
#: this firing entirely rather than dispatch a child onto an unserved queue.
MODE_UNAVAILABLE: Final = "unavailable"


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse an annotation timestamp, treating a naive one as UTC.

    Naive matters: these are hand-editable, and `kubectl annotate` by hand
    rarely includes an offset. Reading a naive stamp as local time would shift
    the fallback window by however far the box is from UTC.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        _log.warning("ignoring unparseable GPU-fallback timestamp %r", raw)
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class GpuState:
    """Fallback bookkeeping, as read from / written to the GPU Deployment.

    The default is "no history": no failed starts recorded, no wedge in
    progress, not in fallback. Every malformed annotation degrades to this,
    which costs at most a few extra GPU attempts and never a crash.
    """

    failures: int = 0
    wedged_since: datetime | None = None
    fallback_until: datetime | None = None

    @classmethod
    def from_annotations(cls, annotations: Mapping[str, str] | None) -> GpuState:
        """Read state out of a Deployment's annotations, leniently.

        These are operator-editable by design, so a typo must not take the
        predict stage down — every unreadable field falls back to its default
        with a warning rather than raising.
        """
        annotations = annotations or {}
        raw_failures = annotations.get(FAILURES_ANNOTATION)
        failures = 0
        if raw_failures:
            try:
                failures = max(0, int(raw_failures))
            except ValueError:
                _log.warning(
                    "ignoring unparseable %s=%r", FAILURES_ANNOTATION, raw_failures
                )
        return cls(
            failures=failures,
            wedged_since=_parse_timestamp(annotations.get(WEDGED_SINCE_ANNOTATION)),
            fallback_until=_parse_timestamp(annotations.get(FALLBACK_UNTIL_ANNOTATION)),
        )

    def to_annotations(self) -> dict[str, str | None]:
        """Annotation patch for this state.

        A ``None`` value **removes** the annotation in a strategic-merge patch,
        which is why an empty state clears all three rather than writing "0"
        and two empty strings: `kubectl describe` on a healthy GPU Deployment
        then shows no fallback bookkeeping at all, so the presence of any of
        these keys is itself the signal that something went wrong.
        """
        return {
            FAILURES_ANNOTATION: str(self.failures) if self.failures else None,
            WEDGED_SINCE_ANNOTATION: _format_timestamp(self.wedged_since),
            FALLBACK_UNTIL_ANNOTATION: _format_timestamp(self.fallback_until),
        }


@dataclass(frozen=True)
class FallbackPolicy:
    """Tunables, resolved from the api-worker's ``[kubernetes]`` config."""

    active_replicas: int = 1
    fallback_replicas: int = 1
    max_start_failures: int = 3
    wedge_grace: timedelta = timedelta(minutes=5)
    fallback_window: timedelta = timedelta(hours=3)


@dataclass(frozen=True)
class GpuDecision:
    """What to do now, and the state to write back."""

    mode: str
    gpu_replicas: int
    fallback_replicas: int
    state: GpuState
    reason: str


def _decide_wedged(
    state: GpuState, *, now: datetime, policy: FallbackPolicy
) -> GpuDecision:
    """The GPU Deployment wants pods and has none Ready. Count it, or flip.

    Split out of `decide` so each function reads as one decision rather than a
    ladder; the ordering here is the whole policy.
    """
    if state.wedged_since is None:
        # First sighting. A pod still pulling its image is not a failed start,
        # so only start the clock.
        return GpuDecision(
            MODE_GPU,
            policy.active_replicas,
            0,
            replace(state, wedged_since=now),
            "GPU worker has no ready pod yet; starting the grace clock",
        )

    if now - state.wedged_since < policy.wedge_grace:
        return GpuDecision(
            MODE_GPU,
            policy.active_replicas,
            0,
            state,
            "GPU worker still starting, inside the grace window",
        )

    failures = state.failures + 1
    if failures >= policy.max_start_failures:
        return GpuDecision(
            MODE_CPU_FALLBACK,
            0,
            policy.fallback_replicas,
            GpuState(
                failures=failures,
                wedged_since=None,
                fallback_until=now + policy.fallback_window,
            ),
            f"GPU worker failed to start {failures} times; "
            f"falling back to CPU inference for {policy.fallback_window}",
        )

    # Restart the clock so a continuous wedge counts once per observation
    # rather than on every poll.
    return GpuDecision(
        MODE_GPU,
        policy.active_replicas,
        0,
        GpuState(failures=failures, wedged_since=now),
        f"GPU worker failed to start ({failures}/{policy.max_start_failures}); "
        "retrying on the GPU",
    )


def decide(
    state: GpuState,
    *,
    now: datetime,
    gpu_ready: bool,
    gpu_wedged: bool,
    policy: FallbackPolicy,
) -> GpuDecision:
    """Choose which Deployment serves the GPU queue, and update the state.

    ``gpu_ready`` / ``gpu_wedged`` come from a single read of the GPU
    Deployment (see `k8s_scaling.readiness`). Both are False for a Deployment
    scaled to 0 — the ordinary cold-start case, which is neither success nor
    failure.
    """
    if state.fallback_until is not None:
        if now < state.fallback_until:
            return GpuDecision(
                MODE_CPU_FALLBACK,
                0,
                policy.fallback_replicas,
                state,
                f"in the CPU fallback window until {_format_timestamp(state.fallback_until)}",
            )
        # Expired. Clean slate, so the GPU gets a full budget of attempts
        # again — a transient NRP shortage must not strand us on CPU forever.
        return GpuDecision(
            MODE_GPU,
            policy.active_replicas,
            0,
            GpuState(),
            "CPU fallback window expired; probing the GPU worker again",
        )

    if gpu_ready:
        return GpuDecision(
            MODE_GPU,
            policy.active_replicas,
            0,
            GpuState(),
            "GPU worker is ready",
        )

    if gpu_wedged:
        return _decide_wedged(state, now=now, policy=policy)

    # Scaled to zero, or scaled up with pods still being counted: neither a
    # success nor a failure. Scale up and leave the bookkeeping alone.
    return GpuDecision(
        MODE_GPU,
        policy.active_replicas,
        0,
        state,
        "scaling the GPU worker up",
    )

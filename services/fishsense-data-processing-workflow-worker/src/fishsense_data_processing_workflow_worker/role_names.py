"""Role vocabulary: the values `general.role` accepts, and their task queues.

Deliberately a leaf module — it imports nothing from this package. `config.py`
validates `general.role` against `ROLES`, and `roles.py` (which imports every
activity and workflow to build the registration lists) imports these names and
re-exports them. Putting the constants in `roles.py` itself is the obvious
thing and it does not work: `config` -> `roles` -> `activities.utils` ->
`config` is a cycle, and it fails at worker startup rather than at test time.

Three kinds of work, on three queues:

* ``cpu`` — the per-image fan-out stages: rectify/overlay/JPEG for 0.1 / 2 /
  5.1 / 9. Each decodes a full-res `.ORF` and peaks at 1-3 GB, which is why
  this role runs `max_concurrent_activities = 2`.
* ``gpu`` — the two torch stages, `predict_laser_image` and the retired
  `predict_slate_image`.
* ``light`` — the stages that hold no image bytes: frame clustering, laser
  calibration, fish measurement, laser depth, laser-label validation and the
  auto-accept gate. Rows in, numpy, rows out.
* ``all`` — every role in one process. Not a fourth kind of work: it is what
  the devcontainer and the integration tests run so a local process still
  serves every queue the api-worker dispatches to.

``light`` is split out of ``cpu`` for memory, not for CPU. Both are CPU-only
and could share a pod; what they cannot share is the concurrency cap, because
the decoders' cap is bounded by pod memory and a sub-second line fit then
inherits it. Two slots meant one 34-image preprocess dispatch owned the queue,
and on 2026-09-04 that expired three of the auto-accept drain's first four
firings and two consecutive laser calibrations. See `task_queues.py` for the
incident history behind the cap.
"""

from typing import Final

from fishsense_shared import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_LIGHT_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)

ROLE_CPU: Final = "cpu"
ROLE_GPU: Final = "gpu"
ROLE_LIGHT: Final = "light"
ROLE_ALL: Final = "all"

#: Every value `general.role` accepts.
ROLES: Final = (ROLE_CPU, ROLE_GPU, ROLE_LIGHT, ROLE_ALL)

#: Single-role task queues. ``all`` is absent on purpose — it is every queue.
ROLE_TASK_QUEUES: Final[dict[str, str]] = {
    ROLE_CPU: DATA_PROCESSING_TASK_QUEUE,
    ROLE_GPU: DATA_PROCESSING_GPU_TASK_QUEUE,
    ROLE_LIGHT: DATA_PROCESSING_LIGHT_TASK_QUEUE,
}

__all__ = [
    "ROLES",
    "ROLE_ALL",
    "ROLE_CPU",
    "ROLE_GPU",
    "ROLE_LIGHT",
    "ROLE_TASK_QUEUES",
]

"""Role vocabulary: the values `general.role` accepts, and their task queues.

Deliberately a leaf module — it imports nothing from this package. `config.py`
validates `general.role` against `ROLES`, and `roles.py` (which imports every
activity and workflow to build the registration lists) imports these names and
re-exports them. Putting the constants in `roles.py` itself is the obvious
thing and it does not work: `config` -> `roles` -> `activities.utils` ->
`config` is a cycle, and it fails at worker startup rather than at test time.

Split by whether the work needs a GPU:

* ``cpu`` — rectify/overlay/JPEG (stages 0.1 / 2 / 5.1 / 9), frame clustering,
  laser calibration, fish measurement, laser depth, laser-label validation.
* ``gpu`` — the two torch stages, `predict_laser_image` and the retired
  `predict_slate_image`.
* ``all`` — both roles in one process. Not a third kind of work: it is what
  the devcontainer and the integration tests run so a local process still
  serves every queue the api-worker dispatches to.
"""

from typing import Final

from fishsense_shared import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)

ROLE_CPU: Final = "cpu"
ROLE_GPU: Final = "gpu"
ROLE_ALL: Final = "all"

#: Every value `general.role` accepts.
ROLES: Final = (ROLE_CPU, ROLE_GPU, ROLE_ALL)

#: Single-role task queues. ``all`` is absent on purpose — it is two queues.
ROLE_TASK_QUEUES: Final[dict[str, str]] = {
    ROLE_CPU: DATA_PROCESSING_TASK_QUEUE,
    ROLE_GPU: DATA_PROCESSING_GPU_TASK_QUEUE,
}

__all__ = [
    "ROLES",
    "ROLE_ALL",
    "ROLE_CPU",
    "ROLE_GPU",
    "ROLE_TASK_QUEUES",
]

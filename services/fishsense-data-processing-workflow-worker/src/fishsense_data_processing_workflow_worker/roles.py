"""What each data-worker role registers, and on which task queue.

The data-worker runs in one of two roles, and the split is by whether the work
needs a GPU:

* ``cpu`` — the nine stages that are pure CPU: rectify/overlay/JPEG for stages
  0.1 / 2 / 5.1 / 9, frame clustering, laser calibration, fish measurement,
  laser depth, and laser-label validation.
* ``gpu`` — the two torch inference stages: `predict_laser_image`
  (`fishsense_core.laser.LaserDetector`) and the retired `predict_slate_image`
  (`fishsense_core.slate.BoardMasker`).

The GPU queue means **prefer a GPU, not require one**, and that distinction is
what makes it the right home for both. It is served by a GPU Deployment *or*,
when that repeatedly fails to start, a CPU-only one running the same
checkpoints — so landing a stage here buys it a GPU when there is one without
ever gating it on there being one.

The two stages want that for different reasons. The laser detector genuinely
needs the GPU; it is slow enough on CPU that the fallback is a degradation.
The slate masker does not — fishsense-core measures it at 202 ms/frame on CPU
and defaults `BoardMasker(device=)` to ``"cpu"`` for exactly that reason — but
it is still a torch model that goes faster on a card that is already there for
the laser stage, and it costs nothing to let it use one. `predict_slate_image`
now passes a device explicitly; before that it ran on the CPU even on the
GPU-pinned pod, which nothing in the code said out loud.

Before the split there was one process, one queue, and one Deployment that
requested ``nvidia.com/gpu: 1`` with node affinity pinned to compute capability
>= 7.5. That meant *every* stage waited on NRP finding a Turing-or-newer GPU
with free capacity, and the GPU then sat idle through the hours of rawpy decode
that dominate a real dive. Splitting the roles is what lets the CPU Deployment
carry no GPU request at all.

``all`` is not a third role — it is "both roles in one process", which is what
the devcontainer and the integration tests use so a local run still serves
every queue the api-worker dispatches to.

**Keeping the two lists exhaustive is the whole contract here**, and
`tests/test_worker_roles.py` asserts it, because both ways of getting it wrong
are silent. An activity in neither list is registered nowhere: its workflow's
`execute_activity` sits pending until schedule-to-close, hours later, with
nothing logged by either worker. An activity in both is registered on a queue
whose pod may have no GPU at all. Neither surfaces until a dive is being
processed in prod. Add a new activity to exactly one list.
"""

from __future__ import annotations

from typing import Any, Callable, Final, NamedTuple, Sequence

from fishsense_data_processing_workflow_worker.role_names import (
    ROLE_ALL,
    ROLE_CPU,
    ROLE_GPU,
    ROLE_LIGHT,
    ROLE_TASK_QUEUES,
    ROLES,
)

from fishsense_data_processing_workflow_worker.activities.cluster_dive_frames import (
    cluster_dive_frames,
)
from fishsense_data_processing_workflow_worker.activities.compute_laser_depths_activity import (  # noqa: E501  pylint: disable=line-too-long
    compute_laser_depths_activity,
)
from fishsense_data_processing_workflow_worker.activities.evaluate_laser_auto_accept_activity import (  # noqa: E501  pylint: disable=line-too-long
    evaluate_laser_auto_accept_activity,
)
from fishsense_data_processing_workflow_worker.activities.measure_fish_activity import (
    measure_fish_activity,
)
from fishsense_data_processing_workflow_worker.activities.perform_laser_calibration_activity import (  # noqa: E501  pylint: disable=line-too-long
    perform_laser_calibration_activity,
)
from fishsense_data_processing_workflow_worker.activities.predict_laser_image import (
    predict_laser_image,
)
from fishsense_data_processing_workflow_worker.activities.predict_slate_image import (
    predict_slate_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_headtail_image import (  # noqa: E501  pylint: disable=line-too-long
    preprocess_headtail_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_laser_image import (
    preprocess_laser_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_slate_image import (
    preprocess_slate_image,
)
from fishsense_data_processing_workflow_worker.activities.preprocess_species_image import (  # noqa: E501  pylint: disable=line-too-long
    preprocess_species_image,
)
from fishsense_data_processing_workflow_worker.activities.validate_laser_labels_for_dive_activity import (  # noqa: E501  pylint: disable=line-too-long
    validate_laser_labels_for_dive_activity,
)
from fishsense_data_processing_workflow_worker.workflows.compute_laser_depths_workflow import (  # noqa: E501  pylint: disable=line-too-long
    ComputeLaserDepthsWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.dive_frame_clustering_workflow import (  # noqa: E501  pylint: disable=line-too-long
    DiveFrameClusteringWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.measure_fish_workflow import (
    MeasureFishWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.perform_laser_calibration_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PerformLaserCalibrationWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.predict_laser_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictLaserImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.predict_slate_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PredictSlateImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.evaluate_laser_auto_accept_workflow import (  # noqa: E501  pylint: disable=line-too-long
    EvaluateLaserAutoAcceptWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_headtail_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessHeadtailImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_laser_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessLaserImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_slate_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessSlateImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.preprocess_species_images_workflow import (  # noqa: E501  pylint: disable=line-too-long
    PreprocessSpeciesImagesWorkflow,
)
from fishsense_data_processing_workflow_worker.workflows.validate_laser_labels_for_dive_workflow import (  # noqa: E501  pylint: disable=line-too-long
    ValidateLaserLabelsForDiveWorkflow,
)

# The per-image fan-out stages, and only those. Every one of these decodes a
# full-res `.ORF` with rawpy and peaks at 1-3 GB, which is why this role runs
# `general.max_concurrent_activities = 2` and why the pod's memory limit is
# derived from that number. Adding anything cheap here puts it behind those
# two slots — which is exactly the bug the light role exists to fix.
CPU_WORKFLOWS: Final[Sequence[type]] = (
    PreprocessHeadtailImagesWorkflow,
    PreprocessLaserImagesWorkflow,
    PreprocessSlateImagesWorkflow,
    PreprocessSpeciesImagesWorkflow,
)

CPU_ACTIVITIES: Final[Sequence[Callable[..., Any]]] = (
    preprocess_headtail_image,
    preprocess_laser_image,
    preprocess_slate_image,
    preprocess_species_image,
)

# The stages that hold no image bytes: they fetch rows from fishsense-api, do
# numpy, and write rows back. Flat memory, sub-second to seconds of work, no
# object-store traffic and no fan-out — so this pod can be small and can run a
# much higher concurrency than the decoders'.
#
# They were all on the CPU queue until 2026-09-04, when the cost of that became
# measurable: three of the auto-accept drain's first four firings and two
# consecutive laser calibrations died with `ScheduleToStart timeout`, waiting
# behind a 34-image preprocess dispatch for work that takes under a second.
LIGHT_WORKFLOWS: Final[Sequence[type]] = (
    ComputeLaserDepthsWorkflow,
    DiveFrameClusteringWorkflow,
    EvaluateLaserAutoAcceptWorkflow,
    MeasureFishWorkflow,
    PerformLaserCalibrationWorkflow,
    ValidateLaserLabelsForDiveWorkflow,
)

LIGHT_ACTIVITIES: Final[Sequence[Callable[..., Any]]] = (
    cluster_dive_frames,
    compute_laser_depths_activity,
    evaluate_laser_auto_accept_activity,
    measure_fish_activity,
    perform_laser_calibration_activity,
    validate_laser_labels_for_dive_activity,
)

# The torch inference stages. `predict_slate_image` is RETIRED (2026-08-03 —
# the ECC gate does not transfer out of distribution) and its schedule is
# actively deleted at api-worker startup, but it stays registered so a future
# evaluation can start it by hand, and it should get a GPU when one is there.
GPU_WORKFLOWS: Final[Sequence[type]] = (
    PredictLaserImagesWorkflow,
    PredictSlateImagesWorkflow,
)

GPU_ACTIVITIES: Final[Sequence[Callable[..., Any]]] = (
    predict_laser_image,
    predict_slate_image,
)

#: The union, for the exhaustiveness tripwire in `tests/test_worker_roles.py`.
ALL_WORKFLOWS: Final[Sequence[type]] = (
    *CPU_WORKFLOWS,
    *GPU_WORKFLOWS,
    *LIGHT_WORKFLOWS,
)
ALL_ACTIVITIES: Final[Sequence[Callable[..., Any]]] = (
    *CPU_ACTIVITIES,
    *GPU_ACTIVITIES,
    *LIGHT_ACTIVITIES,
)


class Registration(NamedTuple):
    """One Temporal worker's worth of wiring: a queue and what serves it."""

    task_queue: str
    workflows: Sequence[type]
    activities: Sequence[Callable[..., Any]]


_REGISTRATIONS: Final[dict[str, Registration]] = {
    ROLE_CPU: Registration(ROLE_TASK_QUEUES[ROLE_CPU], CPU_WORKFLOWS, CPU_ACTIVITIES),
    ROLE_GPU: Registration(ROLE_TASK_QUEUES[ROLE_GPU], GPU_WORKFLOWS, GPU_ACTIVITIES),
    ROLE_LIGHT: Registration(
        ROLE_TASK_QUEUES[ROLE_LIGHT], LIGHT_WORKFLOWS, LIGHT_ACTIVITIES
    ),
}


def registration_for_role(role: str) -> Registration:
    """Return the single role's wiring.

    Raises for ``all`` as well as for an unknown role: ``all`` is two queues,
    so there is no one answer, and quietly returning the CPU half would leave
    the GPU queue unserved — a stall with no error anywhere.
    """
    try:
        return _REGISTRATIONS[role]
    except KeyError:
        raise ValueError(
            f"{role!r} is not a single data-worker role; "
            f"expected one of {sorted(_REGISTRATIONS)} "
            f"(use {ROLE_ALL!r} with build_workers to serve both queues)"
        ) from None


def registrations_for_role(role: str) -> list[Registration]:
    """Return every worker `role` should run — two entries for ``all``."""
    if role == ROLE_ALL:
        return [
            _REGISTRATIONS[ROLE_CPU],
            _REGISTRATIONS[ROLE_GPU],
            _REGISTRATIONS[ROLE_LIGHT],
        ]
    return [registration_for_role(role)]


def queue_for_role(role: str) -> str:
    """Task queue a single role polls."""
    return registration_for_role(role).task_queue


# Re-exported from `role_names` so callers have one import for "roles": the
# vocabulary and the wiring. `config.py` must keep importing the leaf module
# directly — importing this one would cycle back through the activities.
__all__ = [
    "ALL_ACTIVITIES",
    "ALL_WORKFLOWS",
    "CPU_ACTIVITIES",
    "CPU_WORKFLOWS",
    "GPU_ACTIVITIES",
    "GPU_WORKFLOWS",
    "LIGHT_ACTIVITIES",
    "LIGHT_WORKFLOWS",
    "ROLES",
    "ROLE_ALL",
    "ROLE_CPU",
    "ROLE_GPU",
    "ROLE_LIGHT",
    "Registration",
    "queue_for_role",
    "registration_for_role",
    "registrations_for_role",
]

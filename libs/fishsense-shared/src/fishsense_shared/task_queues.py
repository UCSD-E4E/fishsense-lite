"""Temporal task-queue names shared by the api-worker and the data-worker.

These live here for the same reason `preprocess_contracts.py` and
`object_store.py` do: a task-queue name is an agreement *between* the two
workers, so neither owns it. The api-worker names the queue when it dispatches
a child workflow; the data-worker names the queue it polls. A typo on either
side is silent — the child is accepted by the Temporal server and simply sits
`Running` until its execution timeout, hours later, with nothing in either
worker's logs to say why.

Three queues. The first split is whether the work needs a GPU; the second is
whether it decodes images:

* ``DATA_PROCESSING_TASK_QUEUE`` — the per-image fan-out stages:
  rectify/overlay/JPEG for 0.1 / 2 / 5.1 / 9. Each activity decodes a full-res
  `.ORF` and peaks at 1-3 GB.
* ``DATA_PROCESSING_LIGHT_TASK_QUEUE`` — the stages that hold no image bytes:
  frame clustering, laser calibration, fish measurement, laser depth,
  laser-label validation, and the auto-accept gate. Rows in, numpy, rows out.
* ``DATA_PROCESSING_GPU_TASK_QUEUE`` — the two model-inference stages
  (`predict_laser_image`, and the retired `predict_slate_image`), which run a
  torch checkpoint through `fishsense_core.laser.LaserDetector`.

The split exists because the two have opposite scheduling needs. Before it,
one Deployment served everything and requested ``nvidia.com/gpu: 1`` with node
affinity pinned to compute capability >= 7.5 — so *every* stage was gated on
NRP finding a Turing-or-newer GPU with free capacity, and the GPU sat idle
through the hours of rawpy decode that dominate a real dive. Now the CPU
Deployment carries no GPU request at all and cannot be blocked by GPU scarcity.

The GPU queue is deliberately served by **two** Deployments — the GPU one and a
CPU-only fallback that runs the same checkpoint on the CPU when the GPU one
repeatedly fails to start. They share this queue name precisely so that no
workflow has to know which is running; see
`fishsense_api_workflow_worker.activities.gpu_fallback`.

**The light queue exists because of a memory cap, not a CPU one.** The
per-image worker runs `general.max_concurrent_activities = 2`, and that number
is the scar of a real incident: the Temporal default (100) ran ~8 concurrent
rawpy decodes and OOMKilled the pod into CrashLoopBackOff, and a cap of 4 did
it again on 2026-07-21 — 17 restarts, which starved the queue and timed out
the laser-label validation children. So the ceiling cannot simply be raised;
the pod's memory limit is *derived* from it.

Two slots means a single 34-image preprocess dispatch owns the whole queue for
as long as it takes, and everything else waits. On 2026-09-04 that expired
three of the auto-accept drain's first four firings and two consecutive laser
calibration firings, all with `ScheduleToStart timeout`, for work that takes
under a second. Giving the light stages their own queue is what decouples
them: they hold no image bytes, so their pod can be small and their
concurrency cap has nothing to do with the decoders'.
"""

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"
DATA_PROCESSING_GPU_TASK_QUEUE = "fishsense_data_processing_gpu_queue"
DATA_PROCESSING_LIGHT_TASK_QUEUE = "fishsense_data_processing_light_queue"

__all__ = [
    "DATA_PROCESSING_GPU_TASK_QUEUE",
    "DATA_PROCESSING_LIGHT_TASK_QUEUE",
    "DATA_PROCESSING_TASK_QUEUE",
]

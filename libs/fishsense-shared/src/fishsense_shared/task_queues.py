"""Temporal task-queue names shared by the api-worker and the data-worker.

These live here for the same reason `preprocess_contracts.py` and
`object_store.py` do: a task-queue name is an agreement *between* the two
workers, so neither owns it. The api-worker names the queue when it dispatches
a child workflow; the data-worker names the queue it polls. A typo on either
side is silent — the child is accepted by the Temporal server and simply sits
`Running` until its execution timeout, hours later, with nothing in either
worker's logs to say why.

Two queues, split by whether the work needs a GPU:

* ``DATA_PROCESSING_TASK_QUEUE`` — the CPU stages: rectify/overlay/JPEG
  (0.1 / 2 / 5.1 / 9), frame clustering, laser calibration, fish measurement,
  laser depth, laser-label validation. Nine activities, none of which import
  torch.
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
"""

DATA_PROCESSING_TASK_QUEUE = "fishsense_data_processing_queue"
DATA_PROCESSING_GPU_TASK_QUEUE = "fishsense_data_processing_gpu_queue"

__all__ = [
    "DATA_PROCESSING_GPU_TASK_QUEUE",
    "DATA_PROCESSING_TASK_QUEUE",
]

"""Shared internals for the NRP data-worker scale-to-zero activities.

The api-worker is the only thing that knows when there's data-worker
work to do (it dispatches the child workflows), so it owns the
data-worker's replica count: parent workflows scale it up to
``active_replicas`` before dispatching, and an hourly sweeper scales
it back to 0 when the data-worker task queue is quiet. This module
centralizes the bits both activities need — reading the
``[kubernetes]`` config, building a namespaced ``AppsV1Api`` client
from the NRP kubeconfig, and the declarative replica-set call.

Guardrails that live here so "too many pods on NRP" can't happen by
accident:

* Scaling is OFF unless ``kubernetes.kubeconfig_path`` is set — the
  default (local devcontainer, pre-NRP prod) treats the data-worker
  as always-on and these activities no-op.
* ``set_deployment_replicas`` writes an absolute target, never an
  increment — N parents calling it converge on ``active_replicas``,
  pods can't accumulate.
* ``active_replicas`` is clamped to ``[1, MAX_ACTIVE_REPLICAS]`` here,
  on top of the config validator, so a misconfigured value can't ask
  NRP for an arbitrary count.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field
from datetime import timedelta

from fishsense_shared import (
    DATA_PROCESSING_GPU_TASK_QUEUE,
    DATA_PROCESSING_TASK_QUEUE,
)

from fishsense_api_workflow_worker.activities.gpu_fallback import FallbackPolicy
from fishsense_api_workflow_worker.config import settings

# Upper bound on the active-window replica count. >1 is only ever a
# deliberate operator choice (a giant single dive, or resilience on a
# preemption-prone cluster); this caps the blast radius regardless of
# what `kubernetes.active_replicas` is set to.
MAX_ACTIVE_REPLICAS = 4
MIN_ACTIVE_REPLICAS = 1

DEFAULT_DEPLOYMENT_NAME = "fishsense-data-processing-workflow-worker"
# The GPU half of the split worker, and the CPU-only Deployment that serves the
# same `fishsense_data_processing_gpu_queue` when the GPU one cannot start.
# Exactly one of the two is ever scaled up; see `gpu_fallback`.
DEFAULT_GPU_DEPLOYMENT_NAME = f"{DEFAULT_DEPLOYMENT_NAME}-gpu"
DEFAULT_GPU_FALLBACK_DEPLOYMENT_NAME = f"{DEFAULT_GPU_DEPLOYMENT_NAME}-cpu-fallback"

# The CPU fallback runs the same torch checkpoint without a GPU, so it is far
# slower per image and there is no point running several. Capped low
# independently of `active_replicas`, which sizes GPU pods.
MAX_FALLBACK_REPLICAS = 2


@dataclass(frozen=True)
class GpuScalingConfig:
    """Which Deployment serves the GPU queue, and when to stop waiting for it.

    ``deployment_name`` requests a GPU; ``fallback_deployment_name`` does not
    and runs the same torch checkpoint on the CPU. Exactly one is scaled up at
    a time — see `activities.gpu_fallback`.
    """

    deployment_name: str = DEFAULT_GPU_DEPLOYMENT_NAME
    fallback_deployment_name: str = DEFAULT_GPU_FALLBACK_DEPLOYMENT_NAME
    # How long to wait for a pod before calling a start attempt failed. This is
    # what separates "no GPU is available" from "the pod is still pulling its
    # image".
    start_timeout_seconds: int = 600
    policy: FallbackPolicy = field(default_factory=FallbackPolicy)


@dataclass(frozen=True)
class ScalingConfig:
    """Resolved, validated `[kubernetes]` settings for the scaling activities."""

    kubeconfig_path: str
    namespace: str
    deployment_name: str
    active_replicas: int
    idle_cooldown_minutes: int
    # Grouped rather than flattened: these four are one subsystem (which
    # Deployment serves the GPU queue, and when to give up on the GPU one), and
    # they are the only part of the config a caller that just wants the CPU
    # worker never touches. Defaulted so such a caller — or a test — need not
    # restate the GPU topology; `resolve_scaling_config` always sets it fully.
    gpu: GpuScalingConfig = field(default_factory=GpuScalingConfig)

    def sweep_targets(self) -> tuple[tuple[str, str], ...]:
        """Every (deployment, task queue) pair the idle sweeper must consider.

        Three Deployments, two queues: the CPU worker owns
        `fishsense_data_processing_queue`, while the GPU worker and its CPU
        fallback both serve `fishsense_data_processing_gpu_queue` (only one of
        them is up at a time). Returning the pairing here keeps the sweeper
        from having to know the topology.
        """
        return (
            (self.deployment_name, DATA_PROCESSING_TASK_QUEUE),
            (self.gpu.deployment_name, DATA_PROCESSING_GPU_TASK_QUEUE),
            (self.gpu.fallback_deployment_name, DATA_PROCESSING_GPU_TASK_QUEUE),
        )


def resolve_scaling_config() -> ScalingConfig | None:
    """Return the scaling config, or ``None`` when scaling is disabled.

    Disabled = ``kubernetes.kubeconfig_path`` unset (the default — the
    data-worker is assumed always-on). When it *is* set,
    ``kubernetes.namespace`` is required and a clear error is raised if
    it's missing; ``active_replicas`` is clamped to
    ``[MIN_ACTIVE_REPLICAS, MAX_ACTIVE_REPLICAS]``.
    """
    section = settings.get("kubernetes", {}) or {}
    kubeconfig_path = section.get("kubeconfig_path")
    if not kubeconfig_path:
        return None

    namespace = section.get("namespace")
    if not namespace:
        raise ValueError(
            "kubernetes.namespace is required when kubernetes.kubeconfig_path is set"
        )

    deployment_name = section.get("deployment_name") or DEFAULT_DEPLOYMENT_NAME
    active_replicas = max(
        MIN_ACTIVE_REPLICAS,
        min(int(section.get("active_replicas", 1)), MAX_ACTIVE_REPLICAS),
    )
    idle_cooldown_minutes = max(0, int(section.get("idle_cooldown_minutes", 15)))

    # GPU names default off `deployment_name` rather than off the module
    # constant, so overriding the CPU name (a test namespace, a second
    # environment) carries the whole trio with it instead of silently pairing a
    # renamed CPU Deployment with the stock GPU ones.
    gpu_deployment_name = section.get("gpu_deployment_name") or f"{deployment_name}-gpu"
    gpu_fallback_deployment_name = (
        section.get("gpu_fallback_deployment_name")
        or f"{gpu_deployment_name}-cpu-fallback"
    )
    # The grace window must not outlast the start timeout. `ensure_gpu_worker_
    # running_activity` waits out `gpu_start_timeout_seconds` for a pod, then
    # observes; if the grace were longer, that observation would always land
    # inside it, no failure would ever be counted, and the CPU fallback could
    # never trip — the one outcome this whole mechanism exists to prevent.
    gpu_start_timeout_seconds = max(
        0, int(section.get("gpu_start_timeout_seconds", 600))
    )
    wedge_grace_seconds = min(
        max(0, int(section.get("gpu_wedge_grace_minutes", 5))) * 60,
        gpu_start_timeout_seconds,
    )
    # The GPU count is deliberately NOT `active_replicas`. That setting now
    # sizes the CPU worker, where more pods is simply more throughput; here
    # each pod holds a GPU on a contended shared cluster. At 2 the second pod
    # sat Pending on "Insufficient nvidia.com/gpu" (2026-08-20) — requesting
    # and queueing for a card without using it. Defaults to 1, independently.
    gpu_active_replicas = max(
        MIN_ACTIVE_REPLICAS,
        min(int(section.get("gpu_active_replicas", 1)), MAX_ACTIVE_REPLICAS),
    )
    fallback_policy = FallbackPolicy(
        active_replicas=gpu_active_replicas,
        fallback_replicas=max(
            1, min(int(section.get("gpu_fallback_replicas", 1)), MAX_FALLBACK_REPLICAS)
        ),
        # At least 1, or the pipeline would drop to CPU inference on the very
        # first observation and never actually try the GPU.
        max_start_failures=max(1, int(section.get("gpu_max_start_failures", 3))),
        wedge_grace=timedelta(seconds=wedge_grace_seconds),
        fallback_window=timedelta(
            minutes=max(1, int(section.get("gpu_fallback_minutes", 180)))
        ),
    )
    return ScalingConfig(
        kubeconfig_path=kubeconfig_path,
        namespace=namespace,
        deployment_name=deployment_name,
        active_replicas=active_replicas,
        idle_cooldown_minutes=idle_cooldown_minutes,
        gpu=GpuScalingConfig(
            deployment_name=gpu_deployment_name,
            fallback_deployment_name=gpu_fallback_deployment_name,
            start_timeout_seconds=gpu_start_timeout_seconds,
            policy=fallback_policy,
        ),
    )


def apps_v1_api(kubeconfig_path: str):
    """Build an ``AppsV1Api`` bound to the NRP cluster in ``kubeconfig_path``.

    Uses an explicit ``Configuration`` so we don't mutate the
    kubernetes client's global default config (activities can run
    concurrently). Imports the kubernetes client lazily so importing
    this module — which the worker does at startup to register the
    activities — doesn't pull the dependency in until scaling is
    actually used.
    """
    from kubernetes import client as k8s_client, config as k8s_config

    configuration = k8s_client.Configuration()
    k8s_config.load_kube_config(
        config_file=kubeconfig_path, client_configuration=configuration
    )
    api_client = k8s_client.ApiClient(configuration)
    _relax_x509_strict_verification(api_client, configuration)
    return k8s_client.AppsV1Api(api_client)


def _apply_relaxed_verification(ctx: ssl.SSLContext) -> None:
    """Clear OpenSSL 3.x strict mode on ``ctx`` — nothing else.

    Verification stays fully on (``CERT_REQUIRED`` + hostname check); we only
    drop the ``VERIFY_X509_STRICT`` flag that Python 3.13 enabled by default.
    Kept separate so the security-critical invariant is unit-testable.
    """
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT


def _relax_x509_strict_verification(api_client, configuration) -> None:
    """Verify NRP's apiserver cert fully, minus OpenSSL 3.x strict mode.

    Python 3.13 turned on ``ssl.VERIFY_X509_STRICT`` by default, which enforces
    RFC 5280 to the letter — including a mandatory Authority Key Identifier on
    leaf certs. NRP/Nautilus's kubeadm-generated kube-apiserver cert omits AKI,
    so the (otherwise valid) cert is rejected with "Missing Authority Key
    Identifier" and every scale call fails the TLS handshake.

    The kubernetes 36.x client's ``Configuration`` exposes no ``ssl_context``;
    it builds its own strict urllib3 context from ``ca_certs``/``cert_reqs``.
    We swap in a context that keeps full verification — ``CERT_REQUIRED``
    against the pinned cluster CA plus hostname/IP checking — and clears ONLY
    the strict flag. This is emphatically NOT ``insecure-skip-tls-verify``: it
    restores the verification level Python used by default through 3.12.

    No-op when the kubeconfig disables verification (``verify_ssl`` False) —
    there's nothing to relax and we must not silently re-enable it.
    """
    if not (configuration.verify_ssl and configuration.ssl_ca_cert):
        return
    ctx = ssl.create_default_context(cafile=configuration.ssl_ca_cert)
    _apply_relaxed_verification(ctx)
    # Client-cert kubeconfigs (not our token-based one, but stay correct):
    # the context now owns the whole client-side of the handshake.
    if configuration.cert_file and configuration.key_file:
        ctx.load_cert_chain(configuration.cert_file, configuration.key_file)
    # urllib3 gets ambiguous if both ssl_context and ca_certs/cert_reqs are
    # set — hand verification entirely to our context.
    pool_kw = api_client.rest_client.pool_manager.connection_pool_kw
    pool_kw["ssl_context"] = ctx
    for key in ("ca_certs", "cert_reqs", "cert_file", "key_file"):
        pool_kw.pop(key, None)


def deployment_is_wedged(api, namespace: str, name: str) -> bool:
    """True iff the Deployment wants pods but has no Ready one.

    This is the escape hatch on the sweeper's "is the queue busy?" check, and
    it exists because those two signals can disagree in exactly one direction
    that matters. A data-worker that cannot start — expired Temporal cert, bad
    image tag, unschedulable GPU request, exhausted quota — never drains its
    task queue, so every dispatched child sits ``Running`` until it times out
    and the queue never *looks* idle. The sweeper then declines to scale down,
    the Deployment stays pinned at ``active_replicas``, and it holds NRP GPUs
    around the clock while getting no work done. Prod sat exactly there from
    2026-08-14 (see the Temporal-cert notes in CLAUDE.md): the failure
    suppressed the very cleanup that would have bounded its cost.

    "Busy" is only a reason to keep pods alive when the pods can make progress.

    Deliberately shallow — ``spec.replicas`` versus ``status.readyReplicas``,
    both from the one Deployment read the deploy identity is already allowed
    (``deployments: get``; it has no ``pods`` access). Richer signals were
    measured against the live wedge and rejected:

    * ``Progressing`` read ``True``/``NewReplicaSetAvailable`` throughout — the
      ReplicaSet *had* progressed, back when a pod first came up.
    * ``Available``'s ``lastTransitionTime`` is useless as a "wedged since"
      clock. The pod has no readinessProbe, so each crash cycle flips it Ready
      then not-Ready and the timestamp resets every few seconds.

    Nothing here is time-based, which leaves one false positive: a sweep that
    lands inside a genuine cold start (image pull) sees no Ready pod and scales
    down. That costs an hour, not correctness — the next parent with real work
    scales it straight back up, and per-image activities are idempotent by
    design, so the children it interrupts simply re-run.
    """
    return readiness(
        api.read_namespaced_deployment(name=name, namespace=namespace)
    ).wedged


@dataclass(frozen=True)
class Readiness:
    """What one Deployment read says about whether its pods are serving.

    ``ready`` and ``wedged`` are not complements: a Deployment scaled to 0 is
    neither, which is the ordinary state of every data-worker Deployment for
    most of the hour. `gpu_fallback.decide` depends on being able to tell that
    third case apart — counting a scaled-to-zero Deployment as a failed start
    would trip the CPU fallback during normal idle operation.
    """

    desired: int
    ready_count: int

    @property
    def ready(self) -> bool:
        """At least one pod is Ready, so the queue can drain."""
        return self.ready_count > 0

    @property
    def wedged(self) -> bool:
        """Pods are wanted and none is Ready."""
        return self.desired > 0 and self.ready_count == 0


def readiness(deployment) -> Readiness:
    """Read `Readiness` off a Deployment object.

    Takes the object rather than (api, namespace, name) so a caller that also
    needs the annotations — `ensure_gpu_worker_running_activity` does — can
    spend one API read on both instead of racing two.
    """
    desired = (deployment.spec.replicas if deployment.spec else None) or 0
    status = deployment.status
    # k8s OMITS readyReplicas rather than sending 0, so this is None in exactly
    # the wedged case. Reading None as "unknown, assume healthy" is what would
    # keep the GPUs pinned.
    ready_count = (status.ready_replicas if status else None) or 0
    return Readiness(desired=desired, ready_count=ready_count)


def patch_deployment_annotations(
    api, namespace: str, name: str, annotations: dict[str, str | None]
) -> None:
    """Merge annotations onto a Deployment; a ``None`` value removes the key.

    This is a metadata-only strategic-merge patch. It deliberately never
    touches `spec.template`, so it cannot roll the pods — an annotation write
    on the *pod template* would restart the very worker we are trying to get
    running. `tests/test_ensure_gpu_worker_running_activity.py` pins that.
    """
    api.patch_namespaced_deployment(
        name=name,
        namespace=namespace,
        body={"metadata": {"annotations": annotations}},
    )


def set_deployment_replicas(api, namespace: str, name: str, replicas: int) -> None:
    """Set a Deployment's replica count via the scale subresource.

    Idempotent — the count is set absolutely (never added to), and
    patching to the value it already has is a server-side no-op (no
    resourceVersion bump, no pods touched). It always issues the PATCH;
    it doesn't read-then-compare to skip the call.
    """
    api.patch_namespaced_deployment_scale(
        name=name,
        namespace=namespace,
        body={"spec": {"replicas": replicas}},
    )

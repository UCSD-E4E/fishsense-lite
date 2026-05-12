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

from dataclasses import dataclass

from fishsense_api_workflow_worker.config import settings

# Upper bound on the active-window replica count. >1 is only ever a
# deliberate operator choice (a giant single dive, or resilience on a
# preemption-prone cluster); this caps the blast radius regardless of
# what `kubernetes.active_replicas` is set to.
MAX_ACTIVE_REPLICAS = 4
MIN_ACTIVE_REPLICAS = 1

DEFAULT_DEPLOYMENT_NAME = "fishsense-data-processing-workflow-worker"


@dataclass(frozen=True)
class ScalingConfig:
    """Resolved, validated `[kubernetes]` settings for the scaling activities."""

    kubeconfig_path: str
    namespace: str
    deployment_name: str
    active_replicas: int
    idle_cooldown_minutes: int


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
            "kubernetes.namespace is required when kubernetes.kubeconfig_path "
            "is set"
        )

    deployment_name = section.get("deployment_name") or DEFAULT_DEPLOYMENT_NAME
    active_replicas = max(
        MIN_ACTIVE_REPLICAS,
        min(int(section.get("active_replicas", 1)), MAX_ACTIVE_REPLICAS),
    )
    idle_cooldown_minutes = max(0, int(section.get("idle_cooldown_minutes", 15)))
    return ScalingConfig(
        kubeconfig_path=kubeconfig_path,
        namespace=namespace,
        deployment_name=deployment_name,
        active_replicas=active_replicas,
        idle_cooldown_minutes=idle_cooldown_minutes,
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
    # pylint: disable=import-outside-toplevel
    from kubernetes import client as k8s_client, config as k8s_config

    configuration = k8s_client.Configuration()
    k8s_config.load_kube_config(
        config_file=kubeconfig_path, client_configuration=configuration
    )
    return k8s_client.AppsV1Api(k8s_client.ApiClient(configuration))


def set_deployment_replicas(api, namespace: str, name: str, replicas: int) -> None:
    """Declaratively set a Deployment's replica count (idempotent).

    A no-op when it already equals ``replicas``; never adds to the
    current count.
    """
    api.patch_namespaced_deployment_scale(
        name=name,
        namespace=namespace,
        body={"spec": {"replicas": replicas}},
    )

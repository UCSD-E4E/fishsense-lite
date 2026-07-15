"""Unit tests for `temporal_namespace`.

The contract: every worker threads the configured Temporal namespace into
`Client.connect(..., namespace=...)`. krg-infra ADR 0023 warns that OSS
Temporal mTLS authenticates the client but does NOT pin it to a namespace,
so a worker that omits `namespace=` silently lands in `default` instead of
the tenant's `fishsense` namespace. This helper reads it off the standard
`settings.temporal` shape, defaulting to `default` when unset (local dev /
tests) so the behaviour matches temporalio's own default.
"""

from fishsense_shared import temporal_namespace


class _Settings(dict):
    """Minimal Dynaconf-like object: supports `"k" in s` and `s.k`."""

    __getattr__ = dict.__getitem__


def test_returns_configured_namespace():
    assert temporal_namespace(_Settings(namespace="fishsense")) == "fishsense"


def test_defaults_to_default_when_absent():
    # Matches build_tls_config's `"key" in settings` tolerance for optional
    # keys — an un-set namespace must fall back, not raise.
    assert temporal_namespace(_Settings()) == "default"


def test_honours_an_explicit_default_namespace():
    assert temporal_namespace(_Settings(namespace="default")) == "default"

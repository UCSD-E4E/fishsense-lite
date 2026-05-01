"""Tests for the shared URL Dynaconf condition.

Worker `*.url` validators originally used `validators.url`, which
rejects URLs whose hostname has underscores or no TLD — i.e. every
Docker-internal hostname (`static_file_server`, `fishsense-api`,
`temporal`). The local devcontainer + production compose stacks use
exactly those, so the shared validator is a permissive scheme+host
check that still catches typos like a missing scheme.
"""

from fishsense_shared import url_condition


def test_accepts_docker_internal_hostnames():
    assert url_condition("http://static_file_server")
    assert url_condition("http://fishsense-api:8000")
    assert url_condition("http://temporal:7233")


def test_accepts_public_https_urls():
    assert url_condition("https://orchestrator.fishsense.e4e.ucsd.edu")
    assert url_condition("https://example.com/path?q=1")


def test_accepts_ipv4_urls():
    assert url_condition("http://10.0.0.1")
    assert url_condition("http://10.0.0.1:8080")


def test_rejects_missing_scheme():
    assert not url_condition("static_file_server")
    assert not url_condition("//static_file_server")


def test_rejects_unsupported_scheme():
    assert not url_condition("ftp://static_file_server")
    assert not url_condition("file:///etc/passwd")


def test_rejects_empty_or_garbage():
    assert not url_condition("")
    assert not url_condition("not a url")
    assert not url_condition("http://")

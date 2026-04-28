"""Tests for the URL condition used by the worker's dynaconf settings.

The worker's `*.url` validators originally used `validators.url`, which
rejects URLs whose hostname has underscores or no TLD — i.e. every
Docker-internal hostname (`static_file_server`, `fishsense-api`,
`temporal`). The local devcontainer stack uses exactly those, so the
validator is replaced with a permissive scheme+host check that still
catches typos like a missing scheme.
"""

from fishsense_data_processing_workflow_worker.config import _url_condition


def test_accepts_docker_internal_hostnames():
    assert _url_condition("http://static_file_server")
    assert _url_condition("http://fishsense-api:8000")
    assert _url_condition("http://temporal:7233")


def test_accepts_public_https_urls():
    assert _url_condition("https://orchestrator.fishsense.e4e.ucsd.edu")
    assert _url_condition("https://example.com/path?q=1")


def test_accepts_ipv4_urls():
    assert _url_condition("http://10.0.0.1")
    assert _url_condition("http://10.0.0.1:8080")


def test_rejects_missing_scheme():
    assert not _url_condition("static_file_server")
    assert not _url_condition("//static_file_server")


def test_rejects_unsupported_scheme():
    assert not _url_condition("ftp://static_file_server")
    assert not _url_condition("file:///etc/passwd")


def test_rejects_empty_or_garbage():
    assert not _url_condition("")
    assert not _url_condition("not a url")
    assert not _url_condition("http://")

"""Shared activity helpers for the data-processing worker."""

from fishsense_api_sdk.client import Client

from fishsense_data_processing_workflow_worker.config import settings


def get_fs_client() -> Client:
    """Build a fishsense-api SDK client from the worker's dynaconf settings.

    Centralized so per-activity modules don't each construct one — when
    we eventually add timeouts / retries / custom auth, only this helper
    has to change.
    """
    return Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    )

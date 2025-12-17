"""Utility functions for activities."""

from fishsense_api_sdk.client import Client
from label_studio_sdk.client import LabelStudio

from fishsense_api_workflow_worker.config import settings


def get_ls_client():
    """Get Label Studio client.
    Returns:
        LabelStudio: Label Studio client
    """
    return LabelStudio(
        base_url=settings.label_studio.url, api_key=settings.label_studio.api_key
    )


def get_fs_client() -> Client:
    """Get Fishsense API client.

    Returns:
        Client: Fishsense API client
    """
    return Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    )

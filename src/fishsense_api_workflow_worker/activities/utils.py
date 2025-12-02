"""Utility functions for activities."""

from fishsense_api_sdk.client import Client

from fishsense_api_workflow_worker.config import settings


def get_client() -> Client:
    """Get Fishsense API client.

    Returns:
        Client: Fishsense API client
    """
    return Client(
        settings.fishsense_api.url,
        settings.fishsense_api.username,
        settings.fishsense_api.password,
    )

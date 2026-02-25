"""Utility functions to get various types of labels from the Fishsense API."""

import asyncio
from typing import List

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.laser_label import LaserLabel
from httpx import HTTPStatusError


async def __get_laser_labels(
    fishsense_client: Client, dive_id: int
) -> List[LaserLabel]:
    """Get laser labels for a specific dive from Fishsense API."""
    try:
        return await fishsense_client.labels.get_laser_labels(dive_id)
    except HTTPStatusError as e:
        if e.response.status_code == 404:
            return []

        raise


async def get_laser_labels(fishsense_client: Client) -> List[LaserLabel]:
    """Get all laser labels from Fishsense API."""
    dives = await fishsense_client.dives.get_canonical()

    laser_labels = await asyncio.gather(
        *(__get_laser_labels(fishsense_client, dive.id) for dive in dives)
    )
    laser_labels = [label for sublist in laser_labels for label in sublist]

    return laser_labels


async def __get_species_labels(
    fishsense_client: Client, dive_id: int
) -> List[LaserLabel]:
    """Get species labels for a specific dive from Fishsense API."""
    try:
        return await fishsense_client.labels.get_species_labels(dive_id)
    except HTTPStatusError as e:
        if e.response.status_code == 404:
            return []

        raise


async def get_species_labels(fishsense_client: Client) -> List[LaserLabel]:
    """Get all species labels from Fishsense API."""
    dives = await fishsense_client.dives.get_canonical()

    species_labels = await asyncio.gather(
        *(__get_species_labels(fishsense_client, dive.id) for dive in dives)
    )
    species_labels = [label for sublist in species_labels for label in sublist]

    return species_labels


async def __get_headtail_labels(
    fishsense_client: Client, dive_id: int
) -> List[LaserLabel]:
    """Get head-tail labels for a specific dive from Fishsense API."""
    try:
        return await fishsense_client.labels.get_headtail_labels(dive_id)
    except HTTPStatusError as e:
        if e.response.status_code == 404:
            return []

        raise


async def get_headtail_labels(fishsense_client: Client) -> List[LaserLabel]:
    """Get all head-tail labels from Fishsense API."""
    dives = await fishsense_client.dives.get_canonical()

    headtail_labels = await asyncio.gather(
        *(__get_headtail_labels(fishsense_client, dive.id) for dive in dives)
    )
    headtail_labels = [label for sublist in headtail_labels for label in sublist]

    return headtail_labels


async def __get_dive_slate_labels(
    fishsense_client: Client, dive_id: int
) -> List[LaserLabel]:
    """Get dive slate labels for a specific dive from Fishsense API."""
    try:
        return await fishsense_client.labels.get_dive_slate_labels(dive_id)
    except HTTPStatusError as e:
        if e.response.status_code == 404:
            return []

        raise


async def get_dive_slate_labels(fishsense_client: Client) -> List[LaserLabel]:
    """Get all dive slate labels from Fishsense API."""
    dives = await fishsense_client.dives.get_canonical()

    slate_labels = await asyncio.gather(
        *(__get_dive_slate_labels(fishsense_client, dive.id) for dive in dives)
    )
    slate_labels = [label for sublist in slate_labels for label in sublist]

    return slate_labels

"""Enumeration representing data source types."""

from enum import Enum


class DataSource(str, Enum):
    """Enumeration representing data source types."""

    PREDICTION = "PREDICTION"
    LABEL_STUDIO = "LABEL_STUDIO"

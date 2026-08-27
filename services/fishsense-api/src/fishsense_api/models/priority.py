"""Module defining the Priority enumeration."""

from enum import Enum


class Priority(str, Enum):
    """Enumeration for priority levels."""

    LOW = "LOW"
    HIGH = "HIGH"

    # "Deliberately excluded", as distinct from LOW's "not yet". Every dive is
    # created LOW and promoted by ingest's finalize step, so LOW cannot say
    # anything about intent -- a dive parked forever and a dive waiting its
    # turn look identical. NONE is filtered out by the cohort selectors the
    # same way LOW is; the difference is only legible to a human, which is why
    # it travels with `Dive.notes`.
    NONE = "NONE"

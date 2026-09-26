"""Why a laser label was superseded."""

from enum import Enum


class SupersededReason(str, Enum):
    """Who dead-lettered a `LaserLabel`, recorded on the row that says so.

    NULL means unknown: every supersede before 2026-09-26 carries it, because
    nothing recorded the writer until the laser validator was found to have
    eroded dives a run at a time and the good supersedes could not be told
    from the bad.
    """

    # The per-dive validator's 3-sigma test on a measurement frame.
    VALIDATOR_3SIGMA = "validator_3sigma"
    # The same validator's absolute bound on a calibration (slate) frame.
    VALIDATOR_COARSE_CALIBRATION = "validator_coarse_calibration"
    # An operator, by hand: dive 77's reflection recipe, 347's wild slate dots.
    MANUAL = "manual"
    # The reviewed remediation tool. It *revives* rows, so on a live row this
    # records that the last change to `superseded` was the remediation's.
    REMEDIATION = "remediation"

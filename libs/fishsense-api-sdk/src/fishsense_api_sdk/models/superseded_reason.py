"""Why a laser label was superseded (mirror of the API's enum)."""

from enum import Enum


class SupersededReason(str, Enum):
    """Who dead-lettered a `LaserLabel`. `None` on the row means unknown."""

    VALIDATOR_3SIGMA = "validator_3sigma"
    VALIDATOR_COARSE_CALIBRATION = "validator_coarse_calibration"
    MANUAL = "manual"
    # The remediation tool revives rows; on a live row this records that its
    # last change to `superseded` was the remediation's.
    REMEDIATION = "remediation"

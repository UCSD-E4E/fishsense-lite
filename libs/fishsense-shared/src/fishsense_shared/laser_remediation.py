"""Contract for the laser-supersede remediation run.

Lives here, like `preprocess_contracts`, because it is an agreement between the
operator CLI and api-worker parent (which start and report on a run) and the
data-worker child (which plans and writes). Dataclasses, so Temporal's default
converter carries them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "PARENT_WORKFLOW",
    "CHILD_WORKFLOW",
    "RemediateLaserSupersedesInput",
    "RemediationDiveRequest",
    "revival_digest",
]

PARENT_WORKFLOW = "RemediateLaserSupersedesParentWorkflow"
CHILD_WORKFLOW = "RemediateLaserSupersedesWorkflow"


@dataclass
class RemediateLaserSupersedesInput:
    """One remediation run. Dry run unless `apply` AND a matching digest."""

    dive_ids: List[int]
    excluded_dive_ids: List[int] = field(default_factory=list)
    excluded_label_ids: List[int] = field(default_factory=list)
    apply: bool = False
    # The `plan_sha256` of the reviewed dry-run report. Apply refuses unless
    # the plan it recomputes has this digest.
    expected_plan_sha256: Optional[str] = None


@dataclass
class RemediationDiveRequest:
    """Plan (or apply) one dive. `revive_ids` is read by apply only."""

    dive_id: int
    excluded_label_ids: List[int] = field(default_factory=list)
    dive_excluded: bool = False
    revive_ids: List[int] = field(default_factory=list)


def revival_digest(rows: Iterable[Tuple[int, Sequence[int]]]) -> str:
    """sha256 over exactly the revivals `(dive_id, revive_ids)`, order-free."""
    canonical = sorted(
        (int(dive_id), sorted(int(i) for i in ids)) for dive_id, ids in rows if ids
    )
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()

"""What a parent asks to have un-flagged.

One argument object shared by the four `clear_*_reprocess_flags_activity`
functions, because `_dispatch.run_sdk_activity` passes exactly one argument and
the scope has to travel with the dive id.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ClearReprocessFlagsInput(BaseModel):
    """A dive, and optionally the frames whose flags may come down.

    `checksums=None` means the whole dive. That is the no-work backstop: the
    flag reached no image, so nothing will ever lower it, and leaving it up
    holds the dive in its cohort forever.

    A list — including an empty one — means only these frames. The success path
    passes what it actually redrew, so a flag raised while the data-worker
    child was running (up to two hours) survives to the next firing instead of
    being silently discarded by a run that never acted on it.
    """

    dive_id: int
    checksums: Optional[List[str]] = None

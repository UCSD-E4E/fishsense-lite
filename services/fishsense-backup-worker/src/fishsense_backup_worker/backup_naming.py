"""Pure helpers for backup filename construction and retention pruning.

The naming convention is `{ISO-UTC}.dump` with `:` replaced by `-` so
the filename is portable across filesystems. Lexicographic sort of
these names equals chronological sort, which is the property
`filenames_to_prune` relies on.
"""

import re
from datetime import datetime, timezone
from typing import Iterable, List


# Matches our backup filenames exactly: 2026-04-28T03-00-00Z.dump
_FILENAME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z\.dump$"
)


def backup_filename(when: datetime) -> str:
    """Return the canonical backup filename for a UTC moment.

    Refuses naive datetimes so a UTC/local mix can't sneak into the
    NAS listing. Aware datetimes in non-UTC timezones are normalized
    to UTC.
    """
    if when.tzinfo is None or when.tzinfo.utcoffset(when) is None:
        raise ValueError(
            "backup_filename requires a timezone-aware datetime; "
            "got a naive one"
        )
    utc = when.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H-%M-%SZ.dump")


def filenames_to_prune(filenames: Iterable[str], keep: int) -> List[str]:
    """Return the dump filenames that should be deleted to retain only
    the `keep` most recent.

    Filenames not matching the canonical shape (`{ISO-UTC}.dump`) are
    ignored — they are treated as foreign uploads (stray manual
    backups, READMEs, partial uploads from crashed runs) and left
    alone for a human to triage.
    """
    if keep <= 0:
        raise ValueError(f"keep must be positive, got {keep}")

    candidates = sorted(f for f in filenames if _FILENAME_RE.match(f))
    if len(candidates) <= keep:
        return []
    return candidates[:-keep]

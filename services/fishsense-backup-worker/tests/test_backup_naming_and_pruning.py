"""Pure-logic tests for backup filename + pruning helpers.

The naming convention has to be sortable lexicographically (so a `sort
+ slice` over the NAS file list is enough to identify the most recent
N) and unambiguous about timezone (we always use UTC).
"""

from datetime import datetime, timezone

import pytest

from fishsense_backup_worker.backup_naming import (
    backup_filename,
    filenames_to_prune,
)


# --- backup_filename --------------------------------------------------


def test_backup_filename_uses_iso_utc_timestamp_with_dump_extension():
    moment = datetime(2026, 4, 28, 3, 0, 0, tzinfo=timezone.utc)
    assert backup_filename(moment) == "2026-04-28T03-00-00Z.dump"


def test_backup_filename_normalizes_aware_datetime_to_utc():
    """Pacific 20:00 → UTC 04:00 (during PDT). The filename must stay
    UTC-anchored regardless of the input tz."""
    from datetime import timedelta

    pdt = timezone(timedelta(hours=-7))
    moment = datetime(2026, 4, 27, 21, 0, 0, tzinfo=pdt)
    assert backup_filename(moment) == "2026-04-28T04-00-00Z.dump"


def test_backup_filename_rejects_naive_datetime():
    """Naive datetimes are ambiguous about timezone — refuse them so
    we don't get a UTC/local mix in the NAS listing later."""
    naive = datetime(2026, 4, 28, 3, 0, 0)
    with pytest.raises(ValueError):
        backup_filename(naive)


def test_backup_filenames_sort_chronologically_as_strings():
    """Property the pruning logic depends on: lexicographic sort = time sort."""
    a = backup_filename(datetime(2026, 4, 28, 3, 0, tzinfo=timezone.utc))
    b = backup_filename(datetime(2026, 4, 29, 3, 0, tzinfo=timezone.utc))
    c = backup_filename(datetime(2026, 5, 1, 3, 0, tzinfo=timezone.utc))
    shuffled = [c, a, b]
    assert sorted(shuffled) == [a, b, c]


# --- filenames_to_prune ----------------------------------------------


def _names_for_days(n: int):
    """N daily backup filenames in ascending order."""
    from datetime import timedelta

    base = datetime(2026, 4, 1, 3, 0, tzinfo=timezone.utc)
    return [backup_filename(base + timedelta(days=i)) for i in range(n)]


def test_filenames_to_prune_returns_empty_when_under_limit():
    files = _names_for_days(5)
    assert filenames_to_prune(files, keep=14) == []


def test_filenames_to_prune_returns_empty_at_exact_limit():
    files = _names_for_days(14)
    assert filenames_to_prune(files, keep=14) == []


def test_filenames_to_prune_keeps_most_recent_n_drops_the_rest():
    files = _names_for_days(20)
    to_prune = filenames_to_prune(files, keep=14)
    assert to_prune == files[:6]
    # The ones kept must be the 14 most recent.
    kept = [f for f in files if f not in to_prune]
    assert kept == files[-14:]


def test_filenames_to_prune_handles_unsorted_input():
    """The NAS listing isn't guaranteed sorted; the function must sort
    internally before deciding what's old."""
    files = _names_for_days(20)
    shuffled = list(reversed(files))
    to_prune = filenames_to_prune(shuffled, keep=14)
    assert sorted(to_prune) == files[:6]


def test_filenames_to_prune_ignores_unknown_filenames():
    """Files that don't match the timestamp shape (stray uploads,
    README.md, partial uploads from a crashed previous run) are NOT
    candidates for pruning — leave them alone for a human."""
    files = _names_for_days(20) + ["README.md", "junk.txt"]
    to_prune = filenames_to_prune(files, keep=14)
    assert "README.md" not in to_prune
    assert "junk.txt" not in to_prune
    assert len(to_prune) == 6


def test_filenames_to_prune_rejects_zero_or_negative_keep():
    """A keep=0 would delete everything on next run — almost certainly
    a config typo. Refuse loudly."""
    files = _names_for_days(5)
    with pytest.raises(ValueError):
        filenames_to_prune(files, keep=0)
    with pytest.raises(ValueError):
        filenames_to_prune(files, keep=-1)

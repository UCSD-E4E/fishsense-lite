"""Imported LS tasks carry the fields needed to sort a project by capture time.

Label Studio fixes task order at import: `id` and `inner_id` are assigned then
and are immutable, and re-importing to reorder would mean deleting tasks and
losing their annotations. So a project that imported in the wrong order can
only be rescued by sorting the Data Manager on some column — and until now the
only task data was the image URL, which is named by MD5 checksum and therefore
sorts randomly.

Ordering the label list endpoints (`test_label_lists_are_ordered`) fixes the
import order itself, which is the real repair. These fields are the belt to
that pair of braces: they make the capture order recoverable in the UI even if
some future caller feeds tasks in out of order, and they let an operator
retrofit an already-shuffled project.

`taken` is written as an ISO-8601 string deliberately. Label Studio sorts data
columns as text, so a numeric index sorts "0, 1, 10, 100" — verified against
prod project 287542 before choosing this. ISO-8601 is the format whose text
order equals its chronological order.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


@pytest.fixture(name="build")
def _build():
    from fishsense_api_workflow_worker.activities.populate_utils import (
        build_task_data,
    )

    return build_task_data


class _Image:  # pylint: disable=too-few-public-methods
    def __init__(self, image_id, checksum, taken):
        self.id = image_id
        self.checksum = checksum
        self.taken_datetime = taken


def _image(image_id=7, taken=datetime(2023, 8, 31, 19, 42, 29, tzinfo=timezone.utc)):
    return _Image(image_id, f"{image_id:032d}", taken)


def test_keeps_both_image_keys(build):
    """`image` and `img` both ship — prod labeling configs use either."""
    data = build("preprocess_groups_jpeg", _image())

    assert data["image"] == data["img"]
    assert data["image"].startswith("s3://")
    assert "preprocess_groups_jpeg" in data["image"]


def test_carries_capture_time_as_iso_text(build):
    """Sortable as text, because that is how the Data Manager sorts."""
    data = build("preprocess_groups_jpeg", _image())

    assert data["taken"] == "2023-08-31T19:42:29+00:00"


def test_iso_text_order_matches_chronological_order(build):
    """The property the whole field exists for."""
    early = build("f", _image(1, datetime(2023, 8, 31, 19, 42, 29, tzinfo=timezone.utc)))
    later = build("f", _image(2, datetime(2023, 8, 31, 19, 44, 47, tzinfo=timezone.utc)))
    next_day = build("f", _image(3, datetime(2023, 9, 1, 1, 0, 0, tzinfo=timezone.utc)))

    assert early["taken"] < later["taken"] < next_day["taken"]


def test_carries_image_id(build):
    """Ties within a second are common — EXIF resolution is one second and
    these cameras fire ~4 frames a second — so the id is the tiebreak."""
    data = build("preprocess_groups_jpeg", _image(image_id=4321))

    assert data["image_id"] == 4321


def test_missing_capture_time_is_null_not_absent(build):
    """An image with no EXIF timestamp must still produce a valid task; the
    key stays present so the Data Manager still renders the column."""
    data = build("preprocess_groups_jpeg", _image(taken=None))

    assert data["taken"] is None
    assert data["image"].startswith("s3://")

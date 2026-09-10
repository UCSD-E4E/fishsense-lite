"""The per-dive label list endpoints must return rows in a defined order.

None of the four carried an `ORDER BY`, so Postgres was free to return rows in
physical heap order. That is normally insertion order, which is image-id order,
which is capture order — so for most dives the accident looked like a feature
and nobody noticed.

It stops being an accident the moment rows are rewritten. Prod dive 526 was
carved out of dive 107 by a three-way split on 2026-09-09; every one of its
label rows was UPDATEd, and an UPDATE in Postgres MVCC writes a new tuple at
the end of the heap. The physical order was then unrelated to capture order,
and `populate_species_label_studio_project_activity` faithfully carried that
through: `_select_target_images` appends in the order the API handed it, so
Label Studio assigned `inner_id` — the order "Label All Tasks" walks — to a
shuffle. Measured on the resulting project: Spearman +0.058 against capture
order, versus +0.91 to +1.00 for four sibling dives that were never split.

Ordering by `image_id` rather than by `Image.taken_datetime`: the id is NOT
NULL and unique so the sort is total, while `taken_datetime` is nullable and
has one-second EXIF resolution, which leaves ~4 frames a second tied and needs
a tiebreak anyway. Ingest assigns ids in scan order, so id order is capture
order for any normally-ingested dive.

This is asserted on sqlite, where an unordered SELECT returns insertion order,
so seeding out of order is what makes the test meaningful — it fails without
the ORDER BY.
"""

from __future__ import annotations

import pytest

from tests_support.db import dive, image, reprocess_label_kinds

# Deliberately not ascending: this is the order rows are written in, and
# without an ORDER BY it is the order sqlite hands back.
SEED_ORDER = [40, 10, 30, 20]


def _handler_for(model):
    from fishsense_api.controllers import label_controller as lc
    from fishsense_api.models.dive_slate_label import DiveSlateLabel
    from fishsense_api.models.head_tail_label import HeadTailLabel
    from fishsense_api.models.laser_label import LaserLabel
    from fishsense_api.models.species_label import SpeciesLabel

    return {
        LaserLabel: lc.get_laser_labels_for_dive,
        SpeciesLabel: lc.get_species_labels_for_dive,
        HeadTailLabel: lc.get_headtail_labels_for_dive,
        DiveSlateLabel: lc.get_dive_slate_labels_for_dive,
    }[model]


async def _seed(session, model, image_ids):
    session.add(dive(1))
    for image_id in image_ids:
        session.add(image(image_id, 1))
        session.add(
            model(
                id=None,
                image_id=image_id,
                label_studio_project_id=900,
                label_studio_task_id=image_id,
                completed=False,
                superseded=False,
            )
        )
    await session.flush()


@pytest.mark.parametrize("model", reprocess_label_kinds())
async def test_labels_come_back_in_image_id_order(session, model):
    """Rows written out of order must still be returned ascending."""
    await _seed(session, model, SEED_ORDER)

    labels = await _handler_for(model)(1, session=session)

    got = [label.image_id for label in labels]
    assert got == sorted(SEED_ORDER), (
        f"{model.__name__} returned {got}; an unordered SELECT leaks the "
        "physical row order into Label Studio's task order"
    )


@pytest.mark.parametrize("model", reprocess_label_kinds())
async def test_order_is_stable_across_repeated_reads(session, model):
    """The same query twice must agree — populate runs hourly and a task
    order that drifts between runs would reshuffle a project."""
    await _seed(session, model, SEED_ORDER)

    first = [x.image_id for x in await _handler_for(model)(1, session=session)]
    second = [x.image_id for x in await _handler_for(model)(1, session=session)]

    assert first == second

"""The laser validator must be able to read a dive's superseded labels too.

`validate_laser_labels_for_dive_activity` used to fit and flag only the labels
still live, then supersede what it flagged — and the next hourly run re-fitted
the survivors. `flag_outliers` estimates its noise scale from the rows it is
handed, so each pass over a narrower population flagged more. Prod dive 521
lost 15, then 7, then 1 label on three consecutive runs with no label edited in
between; replaying the fit reproduces those exact 23 labels. fishsense-core
#88 now documents the contract: a caller acting on the flags must fit and flag
the FULL population every run, superseded labels included, in a stable order.

So the endpoint grows an opt-in `include_superseded`. The default stays exactly
as it was: every resolver reaches its labels through this endpoint and must
keep not seeing superseded rows.

The order has to be total, not just by image. One image can carry two laser
labels (one per Label Studio project, 461 such images in prod), and
`fit_dive_line`'s RANSAC picks point pairs by row index, so a tie that sqlite
and Postgres break differently would change which line a dive settles on
(core measured dive 257's flag count ranging 41-63 across row orders).
"""

from __future__ import annotations

from tests_support.db import dive, image


async def _seed(session):
    from fishsense_api.models.laser_label import LaserLabel

    session.add(dive(1))
    for image_id in (20, 10):
        session.add(image(image_id, 1))
    # Written out of order, so insertion order is not the answer. Image 10
    # carries two labels (two projects), the tie the secondary key must break.
    rows = [
        (4, 20, 900, False),
        (3, 10, 901, True),
        (2, 10, 900, False),
    ]
    for label_id, image_id, project, superseded in rows:
        session.add(
            LaserLabel(
                id=label_id,
                image_id=image_id,
                label_studio_project_id=project,
                label_studio_task_id=label_id,
                x=1.0,
                y=2.0,
                completed=True,
                superseded=superseded,
            )
        )
    await session.flush()


async def test_default_still_hides_superseded_labels(session):
    from fishsense_api.controllers.label_controller import get_laser_labels_for_dive

    await _seed(session)

    labels = await get_laser_labels_for_dive(1, session=session)

    assert [label.id for label in labels] == [2, 4]


async def test_include_superseded_returns_the_full_population(session):
    from fishsense_api.controllers.label_controller import get_laser_labels_for_dive

    await _seed(session)

    labels = await get_laser_labels_for_dive(
        1, include_superseded=True, session=session
    )

    assert [label.id for label in labels] == [2, 3, 4]
    assert [label.superseded for label in labels] == [False, True, False]


async def test_ties_on_image_are_broken_by_label_id(session):
    """Image 10's two labels were written 3 then 2; they must come back 2, 3."""
    from fishsense_api.controllers.label_controller import get_laser_labels_for_dive

    await _seed(session)

    labels = await get_laser_labels_for_dive(
        1, include_superseded=True, session=session
    )

    assert [(label.image_id, label.id) for label in labels][:2] == [(10, 2), (10, 3)]

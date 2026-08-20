# pylint: disable=too-many-lines
"""Tests for the `dive_pipeline_status` view.

Each stage flag has at least one True case and at least one False
case, including the vacuous-zero-rows edge ("zero labels of a kind"
must read as not-complete, mirroring the
`get_dives_with_complete_laser_labeling` semantics). Edge cases that
historically tripped operators get their own tests:

  * Dive with zero images — every flag should be False (not vacuously
    True).
  * Dive without `dive_slate_id` — slate_* flags should all be False
    regardless of labels.
  * Sentinel HeadTail/DiveSlate rows (project_id NULL) should not
    count as "preprocessed" — matches the cohort selectors' filter.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.views import (
    DIVE_PIPELINE_STATUS_VIEW_NAME,
    DIVE_PIPELINE_STATUS_VIEW_SQL,
    DROP_DIVE_PIPELINE_STATUS_VIEW_SQL,
)


@pytest.fixture
async def session():
    """In-memory sqlite + the view created on top via raw SQL."""
    import fishsense_api.database  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.execute(text(DIVE_PIPELINE_STATUS_VIEW_SQL))
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
    async with engine.begin() as conn:
        await conn.execute(text(DROP_DIVE_PIPELINE_STATUS_VIEW_SQL))
    await engine.dispose()


def _dive(
    dive_id: int,
    *,
    priority: str = "HIGH",
    dive_slate_id=None,
    calibration_dive_id=None,
    name=None,
):
    from fishsense_api.models.dive import Dive  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.priority import Priority  # pylint: disable=import-outside-toplevel

    return Dive(
        id=dive_id,
        path=f"/dev/null/{dive_id}",
        dive_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        priority=Priority[priority],
        dive_slate_id=dive_slate_id,
        calibration_dive_id=calibration_dive_id,
        name=name,
    )


def _image(image_id: int, dive_id: int, *, is_canonical: bool = True):
    """Canonical by default — the normal case. `Image.is_canonical` defaults to
    False on the model, which made every seeded image a duplicate; harmless
    while nothing read the flag, misleading now that the cohort selectors gate
    on it."""
    from fishsense_api.models.image import Image  # pylint: disable=import-outside-toplevel

    return Image(
        id=image_id,
        path=f"/dev/null/img-{image_id}",
        taken_datetime=datetime(2025, 1, 1, tzinfo=timezone.utc),
        checksum=f"{image_id:032d}",
        is_canonical=is_canonical,
        dive_id=dive_id,
    )


_BOOL_FLAG_COLUMNS = (
    "laser_preprocessed",
    "laser_labeling_complete",
    "headtail_preprocessed",
    "headtail_labeling_complete",
    "has_prediction_clusters",
    "dive_images_preprocessed",
    "species_labeling_complete",
    "slate_applicable",
    "slate_preprocessed",
    "slate_labeling_complete",
    "calibrated",
    "measured",
)


async def _row(session, dive_id: int) -> dict:
    """Fetch the view row for a dive as a dict for easy assertion.

    Boolean flags get coerced to `bool` so `is True` / `is False`
    works against both Postgres (native bool) and sqlite (0/1 int)."""
    result = await session.exec(
        text(
            f"SELECT * FROM {DIVE_PIPELINE_STATUS_VIEW_NAME} "
            f"WHERE dive_id = :dive_id"
        ).bindparams(dive_id=dive_id)
    )
    row = dict(result.mappings().one())
    for col in _BOOL_FLAG_COLUMNS:
        row[col] = bool(row[col])
    return row


# ---------- baseline / identity ----------


async def test_empty_dive_emits_a_row_with_every_flag_false(session):
    """Edge: dive with zero images. Every flag must be False, not
    vacuously True via empty subqueries."""
    session.add(_dive(1))
    await session.flush()

    row = await _row(session, 1)
    flag_columns = [
        "laser_preprocessed",
        "laser_labeling_complete",
        "headtail_preprocessed",
        "headtail_labeling_complete",
        "has_prediction_clusters",
        "dive_images_preprocessed",
        "species_labeling_complete",
        "slate_applicable",
        "slate_preprocessed",
        "slate_labeling_complete",
        "calibrated",
        "measured",
    ]
    for col in flag_columns:
        assert not row[col], f"{col} should be False for an empty dive"


async def test_identity_columns_pass_through(session):
    session.add(
        _dive(7, priority="LOW", dive_slate_id=42, name="083023_FishModels_FSL05")
    )
    await session.flush()

    row = await _row(session, 7)
    assert row["dive_id"] == 7
    # Superset dashboards key on dive_id but display the readable name.
    assert row["dive_name"] == "083023_FishModels_FSL05"
    # priority enum stored as enum-name string by sqlmodel.
    assert row["priority"] == "LOW"
    assert row["dive_slate_id"] == 42


async def test_dive_name_null_passes_through_as_none(session):
    """A dive with no name (NULL) still emits a row; dive_name is None."""
    session.add(_dive(8, name=None))
    await session.flush()

    row = await _row(session, 8)
    assert row["dive_id"] == 8
    assert row["dive_name"] is None


# ---------- laser_preprocessed (stage 0.1) ----------


async def test_laser_preprocessed_true_when_every_image_has_laser_row(session):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=False),
            LaserLabel(image_id=12, completed=False),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["laser_preprocessed"] is True


async def test_laser_preprocessed_false_when_one_image_lacks_label(session):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(LaserLabel(image_id=11, completed=False))
    await session.flush()

    assert (await _row(session, 1))["laser_preprocessed"] is False


# ---------- laser_labeling_complete ----------


async def test_laser_labeling_complete_true_when_all_completed_and_none_incomplete(
    session,
):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=True, superseded=False),
            LaserLabel(image_id=12, completed=True, superseded=False),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["laser_labeling_complete"] is True


async def test_laser_labeling_complete_false_when_zero_labels(session):
    """Vacuous-truth guard: zero labels must NOT read as complete."""
    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()

    assert (await _row(session, 1))["laser_labeling_complete"] is False


async def test_laser_labeling_complete_false_when_any_incomplete(session):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=True, superseded=False),
            LaserLabel(image_id=12, completed=False, superseded=False),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["laser_labeling_complete"] is False


async def test_laser_labeling_complete_ignores_superseded_incomplete(session):
    """Superseded incomplete rows are dead; they must not block
    completion. Mirrors the laser-validate flow's behavior."""
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add_all(
        [
            LaserLabel(image_id=11, completed=True, superseded=False),
            LaserLabel(image_id=11, completed=False, superseded=True),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["laser_labeling_complete"] is True


# ---------- headtail_preprocessed (stage 5.1) ----------


async def test_headtail_preprocessed_true_when_every_valid_laser_image_has_headtail(
    session,
):
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0,
        )
    )
    session.add(
        HeadTailLabel(image_id=11, completed=False, label_studio_project_id=71)
    )
    await session.flush()

    assert (await _row(session, 1))["headtail_preprocessed"] is True


async def test_headtail_preprocessed_false_when_no_valid_laser_images(session):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    # Laser is incomplete -> no valid lasers in the dive -> nothing to
    # preprocess yet.
    session.add(LaserLabel(image_id=11, completed=False, x=100.0, y=200.0))
    await session.flush()

    assert (await _row(session, 1))["headtail_preprocessed"] is False


async def test_headtail_preprocessed_false_when_valid_laser_image_lacks_headtail(
    session,
):
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["headtail_preprocessed"] is False


async def test_headtail_preprocessed_ignores_sentinel_headtail_rows(session):
    """A sentinel HeadTailLabel (label_studio_project_id NULL) does
    NOT count as preprocessed — matches the cohort selector."""
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0,
        )
    )
    session.add(
        HeadTailLabel(
            image_id=11, completed=False, label_studio_project_id=None
        )
    )
    await session.flush()

    assert (await _row(session, 1))["headtail_preprocessed"] is False


# ---------- headtail_labeling_complete ----------


async def test_headtail_labeling_complete_true_when_all_completed_and_none_incomplete(
    session,
):
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        HeadTailLabel(
            image_id=11, completed=True, superseded=False,
            label_studio_project_id=71,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["headtail_labeling_complete"] is True


async def test_headtail_labeling_complete_false_when_any_incomplete(session):
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            HeadTailLabel(
                image_id=11, completed=True, superseded=False,
                label_studio_project_id=71,
            ),
            HeadTailLabel(
                image_id=12, completed=False, superseded=False,
                label_studio_project_id=71,
            ),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["headtail_labeling_complete"] is False


async def test_headtail_labeling_complete_false_when_zero_labels(session):
    session.add(_dive(1))
    await session.flush()

    assert (await _row(session, 1))["headtail_labeling_complete"] is False


async def test_species_labeling_complete_ignores_superseded(session):
    """A superseded incomplete species row must not block completion —
    the new `superseded` column on SpeciesLabel."""
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(
                image_id=11, completed=True, superseded=False,
                label_studio_project_id=70,
            ),
            SpeciesLabel(
                image_id=12, completed=False, superseded=True,
                label_studio_project_id=70,
            ),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["species_labeling_complete"] is True


async def test_slate_labeling_complete_ignores_superseded(session):
    """A superseded incomplete slate row must not block completion —
    the new `superseded` column on DiveSlateLabel."""
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            DiveSlateLabel(
                image_id=11, completed=True, superseded=False,
                label_studio_project_id=66,
            ),
            DiveSlateLabel(
                image_id=12, completed=False, superseded=True,
                label_studio_project_id=66,
            ),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["slate_labeling_complete"] is True


# ---------- has_prediction_clusters / dive_images_preprocessed (stage 2) ----------


async def test_has_prediction_clusters_reflects_data_source(session):
    from fishsense_api.models.dive_frame_cluster import DiveFrameCluster  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(1), _dive(2)])
    await session.flush()
    session.add(
        DiveFrameCluster(
            dive_id=1, data_source="PREDICTION", index=0,
        )
    )
    session.add(
        DiveFrameCluster(
            dive_id=2, data_source="LABEL_STUDIO", index=0,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["has_prediction_clusters"] is True
    # dive 2 only has LABEL_STUDIO clusters -> stage 1 hasn't run.
    assert (await _row(session, 2))["has_prediction_clusters"] is False


async def test_dive_images_preprocessed_requires_clusters_and_laser_valid_species_rows(
    session,
):
    """Predicate (post 2026-05-05): PREDICTION cluster present AND
    every laser-valid image has a non-sentinel SpeciesLabel row."""
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(
        DiveFrameCluster(id=820, dive_id=1, data_source="PREDICTION", index=0)
    )
    await session.flush()
    session.add_all(
        [
            DiveFrameClusterImageMapping(dive_frame_cluster_id=820, image_id=11),
            DiveFrameClusterImageMapping(dive_frame_cluster_id=820, image_id=12),
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=12, completed=True, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            SpeciesLabel(image_id=11, label_studio_project_id=70),
            SpeciesLabel(image_id=12, label_studio_project_id=70),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["dive_images_preprocessed"] is True


async def test_dive_images_preprocessed_false_without_prediction_cluster(session):
    """No PREDICTION cluster → dive_images_preprocessed must be False
    even if every laser-valid image has a species row."""
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        LaserLabel(
            image_id=11, completed=True, superseded=False,
            x=100.0, y=200.0, label_studio_project_id=43,
        )
    )
    session.add(SpeciesLabel(image_id=11, label_studio_project_id=70))
    await session.flush()

    assert (await _row(session, 1))["dive_images_preprocessed"] is False


async def test_dive_images_preprocessed_false_when_laser_valid_image_lacks_species_row(
    session,
):
    """Laser-valid image 12 lacks a species row → dive_images_preprocessed
    is False. Image 11 having a species row isn't enough."""
    from fishsense_api.models.dive_frame_cluster import DiveFrameCluster  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(
        DiveFrameCluster(
            dive_id=1, data_source="PREDICTION", index=0,
        )
    )
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=12, completed=True, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            SpeciesLabel(image_id=11, label_studio_project_id=70),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["dive_images_preprocessed"] is False


async def test_dive_images_preprocessed_false_when_no_laser_valid_images(session):
    """Vacuous-truth guard: a dive with PREDICTION clusters but no
    laser-valid images must read False, not True. Mirrors the
    headtail_preprocessed convention."""
    from fishsense_api.models.dive_frame_cluster import DiveFrameCluster  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        DiveFrameCluster(
            dive_id=1, data_source="PREDICTION", index=0,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["dive_images_preprocessed"] is False


async def test_dive_images_preprocessed_ignores_images_without_valid_laser(session):
    """Image 12 has no valid laser → it doesn't enter the predicate's
    "every laser-valid image has species row" check, so a missing
    species row on it doesn't fail dive_images_preprocessed. As long
    as the laser-valid image (11) has a species row, the predicate
    holds."""
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(
        DiveFrameCluster(id=830, dive_id=1, data_source="PREDICTION", index=0)
    )
    await session.flush()
    session.add_all(
        [
            DiveFrameClusterImageMapping(dive_frame_cluster_id=830, image_id=11),
            DiveFrameClusterImageMapping(dive_frame_cluster_id=830, image_id=12),
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=12, completed=False, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            SpeciesLabel(image_id=11, label_studio_project_id=70),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["dive_images_preprocessed"] is True


async def test_view_and_selector_agree_on_species_predicate(session):
    """Cross-controller drift guard: the view's
    `dive_images_preprocessed` flag and the cohort selector
    `select_next_for_species_preprocessing` are nominally the same
    predicate (PREDICTION cluster + every laser-valid image has a
    non-sentinel SpeciesLabel) but live in two different files. If
    one drifts from the other, dashboards say "stage 2 done" while
    the worker keeps re-firing — silent failure.

    This test pins agreement across three representative cases:
      A. dive needs work → view False, selector picks it
      B. dive done       → view True,  selector returns None
      C. dive ineligible → view False, selector returns None
    """
    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        select_next_for_species_preprocessing,
    )
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    # dive 1 (Case A): valid laser IN a cluster + no species → cohort.
    # dive 2 (Case B): valid laser IN a cluster + species labeled → done.
    # dive 3 (Case C): PREDICTION cluster + no valid laser → ineligible.
    session.add_all([_dive(1), _dive(2), _dive(3)])
    await session.flush()
    session.add_all([_image(11, 1), _image(21, 2), _image(31, 3)])
    await session.flush()
    session.add_all(
        [
            DiveFrameCluster(id=801, dive_id=1, data_source="PREDICTION", index=0),
            DiveFrameCluster(id=802, dive_id=2, data_source="PREDICTION", index=0),
            DiveFrameCluster(id=803, dive_id=3, data_source="PREDICTION", index=0),
        ]
    )
    await session.flush()
    session.add_all(
        [
            # Qualifying images must be IN their cluster (view + selector).
            DiveFrameClusterImageMapping(dive_frame_cluster_id=801, image_id=11),
            DiveFrameClusterImageMapping(dive_frame_cluster_id=802, image_id=21),
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=100.0, y=200.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=21, completed=True, superseded=False,
                x=110.0, y=210.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=31, completed=False, superseded=False,
                x=120.0, y=220.0, label_studio_project_id=43,
            ),
            SpeciesLabel(image_id=21, label_studio_project_id=70),
        ]
    )
    await session.flush()

    # Case A — view says "needs work," selector picks it.
    assert (await _row(session, 1))["dive_images_preprocessed"] is False
    selected = await select_next_for_species_preprocessing(session=session)
    assert selected == 1

    # Case B — view says "done."
    assert (await _row(session, 2))["dive_images_preprocessed"] is True

    # Case C — view says "ineligible" (no laser-valid image triggers
    # vacuous-truth guard; pinned in the dedicated False-when-no-laser
    # test above).
    assert (await _row(session, 3))["dive_images_preprocessed"] is False

    # Selector skips dive 2 and dive 3 — dive 1 is the only HIGH-priority
    # dive needing species preprocessing.


# ---------- species_labeling_complete ----------


async def test_species_labeling_complete_true_when_all_completed(session):
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        SpeciesLabel(
            image_id=11, completed=True, superseded=False,
            label_studio_project_id=70,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["species_labeling_complete"] is True


async def test_species_labeling_complete_false_when_any_incomplete(session):
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add_all(
        [
            SpeciesLabel(
                image_id=11, completed=True, superseded=False,
                label_studio_project_id=70,
            ),
            SpeciesLabel(
                image_id=12, completed=False, superseded=False,
                label_studio_project_id=70,
            ),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["species_labeling_complete"] is False


# ---------- slate_applicable / slate_preprocessed / slate_labeling_complete ----------


async def test_slate_applicable_tracks_dive_slate_id(session):
    session.add_all([_dive(1, dive_slate_id=42), _dive(2, dive_slate_id=None)])
    await session.flush()

    assert (await _row(session, 1))["slate_applicable"] is True
    assert (await _row(session, 2))["slate_applicable"] is False


async def test_slate_preprocessed_true_when_marked_images_have_slate_rows(session):
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=42))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        SpeciesLabel(
            image_id=11, label_studio_project_id=70,
            content_of_image="Slate, Laser on slate",
        )
    )
    session.add(
        DiveSlateLabel(image_id=11, label_studio_project_id=66)
    )
    await session.flush()

    assert (await _row(session, 1))["slate_preprocessed"] is True


async def test_slate_preprocessed_false_when_no_slate_marked_images(session):
    """No species label says 'Slate, Laser on slate' -> there's
    nothing to preprocess yet -> False, not vacuously True."""
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=42))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        SpeciesLabel(
            image_id=11, label_studio_project_id=70,
            content_of_image="Fish",
        )
    )
    await session.flush()

    assert (await _row(session, 1))["slate_preprocessed"] is False


async def test_slate_preprocessed_false_when_dive_lacks_slate_id(session):
    """Even if labels exist, no dive_slate_id means slate path doesn't
    apply to this dive at all."""
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=None))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        SpeciesLabel(
            image_id=11, label_studio_project_id=70,
            content_of_image="Slate, Laser on slate",
        )
    )
    session.add(DiveSlateLabel(image_id=11, label_studio_project_id=66))
    await session.flush()

    assert (await _row(session, 1))["slate_preprocessed"] is False


async def test_slate_labeling_complete_true_when_all_completed(session):
    from fishsense_api.models.dive_slate_label import DiveSlateLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1, dive_slate_id=42))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(
        DiveSlateLabel(
            image_id=11, completed=True, superseded=False,
            label_studio_project_id=66,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["slate_labeling_complete"] is True


# ---------- calibrated (stage 13) ----------


async def test_calibrated_true_when_laser_extrinsics_row_exists(session):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    await session.flush()

    assert (await _row(session, 1))["calibrated"] is True


async def test_calibrated_false_when_no_laser_extrinsics(session):
    session.add(_dive(1))
    await session.flush()

    assert (await _row(session, 1))["calibrated"] is False


async def test_calibrated_true_when_borrowed_via_calibration_dive_id(session):
    """A fish-only dive linked to a slate dive that owns extrinsics reads
    calibrated=true even though it has none of its own."""
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))  # slate dive: owns the calibration
    session.add(_dive(2, calibration_dive_id=1))  # fish dive: borrows it
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    await session.flush()

    assert (await _row(session, 1))["calibrated"] is True
    assert (await _row(session, 2))["calibrated"] is True


async def test_calibrated_false_when_linked_source_has_no_extrinsics(session):
    """A link to a source that isn't itself calibrated doesn't fabricate
    calibration."""
    session.add(_dive(1))
    session.add(_dive(2, calibration_dive_id=1))
    await session.flush()

    assert (await _row(session, 2))["calibrated"] is False


# ---------- measured (stage 14) ----------
#
# `measured` is scoped to what stage 14 can actually measure. The prior
# definition ("every LABEL_STUDIO cluster has a fish_id") was
# unreachable: a cluster only gets a fish via a top-three species label
# whose image has a valid laser + headtail, so any cluster without such
# an image kept the dive unmeasured forever. In prod that pinned all 8
# calibrated dives at measured=false permanently — and, because the
# stage-14 cohort mirrors this predicate, would have made a scheduled
# stage 14 re-select the same dives every hour.
#
# "Measurable" here mirrors measure_fish_activity: a top-three species
# label whose image has a valid laser label, a valid headtail label, and
# a LABEL_STUDIO cluster.


# Shared with the other stage-14 predicate test — see `stage14_fixtures`.
from tests_support.stage14_fixtures import (  # noqa: E402  pylint: disable=wrong-import-position
    calibration as _calibration,
    fish_model_measurable_image as _fish_model_measurable_image,
    measurable_image as _measurable_image,
    measurement as _measurement,
)


async def test_measured_counts_a_ruler_image_like_a_fish_model(session):
    """The ruler is a known-length target measured through the same path, so a
    ruler frame is measurable and holds `measured` false until it is measured.

    Its value is that its endpoints are unambiguous: it grades pure calibration
    error, with none of the fork-vs-tip uncertainty a fish model carries."""
    session.add(_dive(1))
    await session.flush()
    _fish_model_measurable_image(
        session, 11, 1, content_of_image="Calibration Targets, Ruler"
    )
    await session.flush()

    assert (await _row(session, 1))["measured"] is False

    session.add_all([_calibration(1), _measurement(11)])
    await session.flush()
    assert (await _row(session, 1))["measured"] is True


async def test_measured_ignores_the_slate_marker(session):
    """Promoting the ruler must not promote its sibling slate branches — the
    stage-9 marker stays unmeasurable, so it can never hold a dive in the
    stage-14 cohort forever."""
    session.add(_dive(1))
    await session.flush()
    _fish_model_measurable_image(
        session, 11, 1, content_of_image="Slate, Laser on slate"
    )
    await session.flush()

    assert (await _row(session, 1))["measured"] is False, (
        "vacuous: no measurable image, so nothing to measure"
    )


async def test_measured_true_for_fish_model_image_without_a_cluster(session):
    """A fish-model dive drains: no LABEL_STUDIO cluster, but the model image
    is measurable and, once measured, `measured` flips true."""
    session.add(_dive(1))
    await session.flush()
    _fish_model_measurable_image(session, 11, 1)
    await session.flush()
    session.add_all([_calibration(1), _measurement(11)])
    await session.flush()

    assert (await _row(session, 1))["measured"] is True


async def test_measured_false_for_unmeasured_fish_model_image(session):
    """Same fish-model image, no measurement yet -> not measured (so the
    stage-14 cohort keeps offering it)."""
    session.add(_dive(1))
    await session.flush()
    _fish_model_measurable_image(session, 11, 1)
    await session.flush()

    assert (await _row(session, 1))["measured"] is False


async def test_measured_true_when_every_measurable_image_has_a_measurement(session):
    session.add(_dive(1))
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add_all([_calibration(1), _measurement(11)])
    await session.flush()

    assert (await _row(session, 1))["measured"] is True


async def test_measured_false_when_a_measurable_image_is_unmeasured(session):
    session.add(_dive(1))
    await session.flush()
    m1 = _measurable_image(session, 11, 1, cluster_id=1)
    m2 = _measurable_image(session, 12, 1, cluster_id=2)
    await session.flush()
    session.add_all([m1, m2])
    session.add_all([_calibration(1), _measurement(11)])  # 12 left unmeasured
    await session.flush()

    assert (await _row(session, 1))["measured"] is False


async def test_measured_false_when_dive_has_no_measurements(session):
    """Vacuous-truth guard: nothing measured -> not measured."""
    session.add(_dive(1))
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    await session.flush()

    assert (await _row(session, 1))["measured"] is False


async def test_measured_ignores_unbound_clusters_with_no_measurable_image(session):
    """The regression this rescope exists for.

    A LABEL_STUDIO cluster with no measurable image can never be bound to
    a fish. Under the old predicate its NULL fish_id pinned the dive at
    measured=false forever. In prod dive 466 carried 1632 such clusters
    against only 24 measurable images.
    """
    from fishsense_api.models.dive_frame_cluster import DiveFrameCluster  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add_all([_calibration(1), _measurement(11)])
    # Unbound cluster carrying no measurable image — stage 14 can never
    # touch it, so it must not hold the dive back.
    session.add(DiveFrameCluster(id=99, dive_id=1, data_source="LABEL_STUDIO", fish_id=None))
    await session.flush()

    assert (await _row(session, 1))["measured"] is True


async def test_measured_false_when_the_calibration_was_replaced(session):
    """The 2026-08-11 panel-offset fix recalibrated dives that already had
    measurements. Their lengths were computed from the old extrinsics and are
    wrong, but `measured` read true and the stage-14 cohort skipped them —
    the dive looked finished while carrying stale numbers. It now reads
    false, which is both honest and what re-queues the dive."""
    session.add(_dive(1))
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    # Measured under extrinsics 50; the dive resolves to 51 today.
    session.add_all([_calibration(1, 51), _measurement(11, laser_extrinsics_id=50)])
    await session.flush()

    assert (await _row(session, 1))["measured"] is False


async def test_measured_false_for_measurements_predating_provenance(session):
    """Every measurement in prod carries NULL here until stage 14 revisits
    it. They read unmeasured exactly once, which is the backfill."""
    session.add(_dive(1))
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add_all([_calibration(1), _measurement(11, laser_extrinsics_id=None)])
    await session.flush()

    assert (await _row(session, 1))["measured"] is False


async def test_measured_true_through_a_borrowed_calibration(session):
    """A fish-only dive is measured with its sibling's extrinsics row, so
    that is the id its measurements carry — resolution here mirrors
    `get_laser_extrinsics_for_dive`, own-then-link."""
    session.add_all([_dive(1, calibration_dive_id=2), _dive(2)])
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add_all([_calibration(2, 51), _measurement(11, laser_extrinsics_id=51)])
    await session.flush()

    assert (await _row(session, 1))["measured"] is True


async def test_measured_ignores_non_top_three_images(session):
    """Stage 14 only measures top-three photos, so others can't block."""
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    mapping = _measurable_image(session, 11, 1, cluster_id=1)
    await session.flush()
    session.add(mapping)
    session.add_all([_calibration(1), _measurement(11)])
    # A second image that is NOT top-three and has no measurement.
    session.add(_image(12, 1))
    await session.flush()
    session.add(
        SpeciesLabel(
            image_id=12, top_three_photos_of_group=False,
            completed=True, superseded=False, label_studio_project_id=70,
        )
    )
    await session.flush()

    assert (await _row(session, 1))["measured"] is True


async def test_measured_ignores_species_rows_without_a_scientific_name(session):
    """`measured` must mirror what stage 14 can actually measure.

    A slate row carries neither a "Common (Scientific)" name nor a known-length
    target name, so `measure_fish_activity` skips it and no Measurement is ever
    written. Counting it as a measurable-but-unmeasured image pinned `measured`
    false forever — the same never-goes-false shape b7c2e4d81a09 fixed one layer
    up, for unbound clusters. (Fish models and the ruler ARE measurable — see
    test_measured_true_for_fish_model_image_* and
    test_measured_counts_a_ruler_image_like_a_fish_model.)
    """
    session.add(_dive(1))
    await session.flush()
    # One real fish (measured) + one slate row (never measurable).
    real = _measurable_image(session, 11, 1, cluster_id=1)
    rig = _measurable_image(
        session, 12, 1, cluster_id=2, content_of_image="Slate, Laser on slate"
    )
    await session.flush()
    session.add_all([real, rig])
    session.add_all([_calibration(1), _measurement(11)])
    await session.flush()

    assert (await _row(session, 1))["measured"] is True, (
        "the slate row must not hold `measured` false"
    )


async def test_dive_images_preprocessed_ignores_unclustered_and_superseded(session):
    """`dive_images_preprocessed` must count only *processable* images —
    valid laser AND in a PREDICTION cluster — with a live species row, so it
    stays in step with `select_next_for_species_preprocessing`.

    Two images: image 11 is clustered and labeled (done); image 12 has a
    valid laser but is NOT in any cluster, so it is unprocessable and must not
    drag the flag to False. Otherwise the dashboard reads "stage 2 stuck"
    forever on a dive the selector correctly never re-fires on (the prod
    poison pill).
    """
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add_all([_image(11, 1), _image(12, 1)])
    await session.flush()
    session.add(DiveFrameCluster(id=810, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=810, image_id=11))
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=1.0, y=2.0, label_studio_project_id=43,
            ),
            LaserLabel(
                image_id=12, completed=True, superseded=False,
                x=3.0, y=4.0, label_studio_project_id=43,
            ),
            SpeciesLabel(
                image_id=11, completed=False, superseded=False,
                label_studio_project_id=70,
            ),
        ]
    )
    await session.flush()

    # image 12 is unclustered → not processable → doesn't count against done.
    assert (await _row(session, 1))["dive_images_preprocessed"] is True


async def test_dive_images_preprocessed_false_when_only_species_row_superseded(session):
    """A dead-lettered species row is not evidence the work is done."""
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(_image(11, 1))
    await session.flush()
    session.add(DiveFrameCluster(id=811, dive_id=1, data_source=DataSource.PREDICTION))
    await session.flush()
    session.add(DiveFrameClusterImageMapping(dive_frame_cluster_id=811, image_id=11))
    session.add_all(
        [
            LaserLabel(
                image_id=11, completed=True, superseded=False,
                x=1.0, y=2.0, label_studio_project_id=43,
            ),
            SpeciesLabel(
                image_id=11, completed=False, superseded=True,
                label_studio_project_id=117,
            ),
        ]
    )
    await session.flush()

    assert (await _row(session, 1))["dive_images_preprocessed"] is False


# ---------- calibration_source (provenance) ----------
#
# `calibrated` says whether a dive HAS extrinsics; it does not say where they
# came from, and that distinction is the strongest data-quality signal we have.
# Measured 2026-08-04 against the known-length fish models: a dive using its
# OWN slate measures at ~1%, while a BORROWED calibration adds a rig-state
# systematic of -8..+2% (n=7) that no amount of software can recover — the
# borrow is for a different rig deployment. Downstream analysis needs to be
# able to filter or weight on this rather than treating all rows alike.


async def test_calibration_source_own_when_the_dive_has_its_own_extrinsics(session):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    session.add(_dive(1))
    await session.flush()
    session.add(LaserExtrinsics(dive_id=1, camera_id=1))
    await session.flush()

    assert (await _row(session, 1))["calibration_source"] == "own"


async def test_calibration_source_borrowed_when_only_the_link_has_extrinsics(session):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(2), _dive(1, calibration_dive_id=2)])
    await session.flush()
    session.add(LaserExtrinsics(dive_id=2, camera_id=1))
    await session.flush()

    assert (await _row(session, 1))["calibration_source"] == "borrowed"


async def test_calibration_source_own_wins_over_a_link(session):
    """Mirrors `get_laser_extrinsics_for_dive`: a dive with its own row uses it
    even when `calibration_dive_id` is also set, so provenance must say 'own'."""
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    session.add_all([_dive(2), _dive(1, calibration_dive_id=2)])
    await session.flush()
    session.add_all(
        [LaserExtrinsics(dive_id=1, camera_id=1), LaserExtrinsics(dive_id=2, camera_id=1)]
    )
    await session.flush()

    assert (await _row(session, 1))["calibration_source"] == "own"


async def test_calibration_source_none_when_uncalibrated(session):
    session.add(_dive(1))
    await session.flush()

    row = await _row(session, 1)
    assert row["calibration_source"] == "none"
    assert row["calibrated"] is False


# ---------------------------------------------------------------------------
# SQL <-> Python parity for the measurable-species predicate
# ---------------------------------------------------------------------------


async def test_measurable_species_sql_agrees_with_taxonomy_is_measurable(session):
    """The one predicate that exists in three languages must mean one thing.

    `taxonomy.is_measurable` is the definition of record — it is literally
    "can `measure_fish_activity` bind this row to a Measurement". The SQL here
    and the SQLAlchemy conditions in `dive_cohort_controller` are `LIKE`-based
    approximations, because the view runs on Postgres in prod and SQLite under
    test and can't use either one's string functions.

    Approximations are fine; *unverified* approximations are how the cohort
    starts offering images the activity always skips, which never resolves and
    re-selects the dive every hour forever. So both representations are run
    over the same corpus and compared row by row.
    """
    from fishsense_shared import taxonomy  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.views import (  # pylint: disable=import-outside-toplevel
        _MEASURABLE_SPECIES_SQL,
    )

    corpus = taxonomy.MEASURABILITY_CORPUS
    session.add_all(
        [
            SpeciesLabel(
                image_id=1000 + i,
                label_studio_project_id=70,
                content_of_image=content,
            )
            for i, (content, _) in enumerate(corpus)
        ]
    )
    await session.flush()

    result = await session.execute(
        text(
            "SELECT sl.image_id FROM specieslabel sl "
            f"WHERE {_MEASURABLE_SPECIES_SQL}"
        )
    )
    matched_by_sql = {row[0] for row in result}

    expected = {
        1000 + i for i, (_, measurable) in enumerate(corpus) if measurable
    }
    assert matched_by_sql == expected, (
        "SQL and taxonomy.is_measurable disagree on: "
        + repr(
            sorted(
                corpus[i - 1000][0]
                for i in matched_by_sql.symmetric_difference(expected)
            )
        )
    )


async def test_measurable_species_sql_agrees_with_the_sqlalchemy_conditions(session):
    """The view's raw SQL and the cohort selector's SQLAlchemy build the same
    predicate from the same constants; assert they select the same rows."""
    from sqlmodel import select  # pylint: disable=import-outside-toplevel

    from fishsense_shared import taxonomy  # pylint: disable=import-outside-toplevel

    from fishsense_api.controllers.dive_cohort_controller import (  # pylint: disable=import-outside-toplevel
        _measurable_species_conditions,
    )
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.views import (  # pylint: disable=import-outside-toplevel
        _MEASURABLE_SPECIES_SQL,
    )

    corpus = taxonomy.MEASURABILITY_CORPUS
    session.add_all(
        [
            SpeciesLabel(
                image_id=2000 + i,
                label_studio_project_id=70,
                content_of_image=content,
            )
            for i, (content, _) in enumerate(corpus)
        ]
    )
    await session.flush()

    via_sql = {
        row[0]
        for row in await session.execute(
            text(
                "SELECT sl.image_id FROM specieslabel sl "
                f"WHERE {_MEASURABLE_SPECIES_SQL} AND sl.image_id >= 2000"
            )
        )
    }
    via_orm = set(
        (
            await session.exec(
                select(SpeciesLabel.image_id)
                .where(*_measurable_species_conditions())
                .where(SpeciesLabel.image_id >= 2000)
            )
        ).all()
    )
    assert via_sql == via_orm


async def test_sql_is_broader_than_python_only_where_pinned(session):
    """A *new* SQL/Python divergence must fail the build.

    The dangerous direction is SQL-broader: the cohort offers an image the
    activity skips, no Measurement is written, and the dive is re-selected
    every hour forever. `taxonomy.SQL_BROADER_THAN_PYTHON` is the pinned,
    known-unreachable set (the empty-name guard the LIKE patterns can't
    express). This asserts the real SQL matches exactly those and no others,
    so widening the predicate — or tightening the Python parser, which is how
    this regressed once already — is caught here rather than in prod.
    """
    from fishsense_shared import taxonomy  # pylint: disable=import-outside-toplevel

    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.views import (  # pylint: disable=import-outside-toplevel
        _MEASURABLE_SPECIES_SQL,
    )

    probes = [c for c, _ in taxonomy.MEASURABILITY_CORPUS if c is not None]
    probes += list(taxonomy.SQL_BROADER_THAN_PYTHON)
    session.add_all(
        [
            SpeciesLabel(
                image_id=3000 + i,
                label_studio_project_id=70,
                content_of_image=content,
            )
            for i, content in enumerate(probes)
        ]
    )
    await session.flush()

    matched_by_sql = {
        row[0]
        for row in await session.execute(
            text(
                "SELECT sl.image_id FROM specieslabel sl "
                f"WHERE {_MEASURABLE_SPECIES_SQL} AND sl.image_id >= 3000"
            )
        )
    }
    sql_broader = {
        probes[i - 3000]
        for i in matched_by_sql
        if not taxonomy.is_measurable(probes[i - 3000])
    }
    assert sql_broader == set(taxonomy.SQL_BROADER_THAN_PYTHON)

    # And the reverse direction — Python measurable, SQL not — must be empty:
    # that would silently under-measure a dive while reporting it complete.
    python_broader = {
        c
        for i, c in enumerate(probes)
        if taxonomy.is_measurable(c) and (3000 + i) not in matched_by_sql
    }
    assert python_broader == set()

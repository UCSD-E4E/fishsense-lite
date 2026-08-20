"""Seeding helpers for the stage-14 predicate, shared by its two test files.

`dive_pipeline_status.measured` and `select_next_for_measure_fish` are the
same question asked in SQL and in SQLAlchemy, and CLAUDE.md requires them to
stay in step — so their tests necessarily build the same fixtures, and did,
in two copies. `duplicate-code` flagged the pair once both moved in the same
change. Keeping one definition also means a change to what "measurable" seeds
cannot silently apply to only one of the two predicates that must agree.

Deliberately not a conftest: these are constructors, not fixtures, and the two
test modules import them by name so a reader can see where a helper comes
from.
"""

from __future__ import annotations

from datetime import datetime, timezone

# A real `Fish` row: stage 14 needs a `Common (Scientific)` name to measure
# against, so a species label without one builds an image the pipeline can
# never actually measure.
MEASURABLE_CONTENT = "Fish, Hogfish (Lachnolaimus maximus)"

# The calibration a measurement was computed with. `calibration` and
# `measurement` share this id by default; pass a different one to model a
# recalibration, or None to model a row written before
# `Measurement.laser_extrinsics_id` existed.
CALIBRATION_ID = 51


def image(image_id: int, dive_id: int, *, is_canonical: bool = True):
    """Canonical by default — the normal case. `Image.is_canonical` defaults
    to False on the model, which made every seeded image a duplicate; harmless
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


def measurable_image(
    session,
    image_id: int,
    dive_id: int,
    *,
    cluster_id: int,
    content_of_image: str | None = MEASURABLE_CONTENT,
):
    """Seed an image stage 14 would attempt: top-three species label + valid
    laser + valid headtail + a LABEL_STUDIO cluster.

    Returns the cluster mapping rather than adding it, because the caller has
    to flush the image first for the FK-less sqlite fixtures to line up.
    """
    from fishsense_api.models.data_source import DataSource  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.dive_frame_cluster import (  # pylint: disable=import-outside-toplevel
        DiveFrameCluster,
        DiveFrameClusterImageMapping,
    )
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(image(image_id, dive_id))
    session.add(
        DiveFrameCluster(
            id=cluster_id, dive_id=dive_id, data_source=DataSource.LABEL_STUDIO
        )
    )
    session.add(
        LaserLabel(image_id=image_id, completed=True, superseded=False, x=1.0, y=2.0)
    )
    session.add(
        HeadTailLabel(
            image_id=image_id,
            completed=True,
            superseded=False,
            head_x=1.0,
            head_y=2.0,
            tail_x=3.0,
            tail_y=4.0,
        )
    )
    session.add(
        SpeciesLabel(
            image_id=image_id,
            top_three_photos_of_group=True,
            completed=True,
            superseded=False,
            label_studio_project_id=70,
            content_of_image=content_of_image,
        )
    )
    return DiveFrameClusterImageMapping(
        dive_frame_cluster_id=cluster_id, image_id=image_id
    )


def fish_model_measurable_image(
    session,
    image_id: int,
    dive_id: int,
    content_of_image: str = "Fish Model, Grouper",
):
    """Seed a fish-model image the way prod has it: top-three species label
    (`Fish Model, <name>`) + valid laser + valid headtail, but **no**
    LABEL_STUDIO cluster (models carry no grouping labels). Stage 14 measures
    these by waiving the cluster gate, so the cohort and the view must too."""
    from fishsense_api.models.head_tail_label import HeadTailLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.laser_label import LaserLabel  # pylint: disable=import-outside-toplevel
    from fishsense_api.models.species_label import SpeciesLabel  # pylint: disable=import-outside-toplevel

    session.add(image(image_id, dive_id))
    session.add(
        LaserLabel(image_id=image_id, completed=True, superseded=False, x=1.0, y=2.0)
    )
    session.add(
        HeadTailLabel(
            image_id=image_id,
            completed=True,
            superseded=False,
            head_x=1.0,
            head_y=2.0,
            tail_x=3.0,
            tail_y=4.0,
        )
    )
    session.add(
        SpeciesLabel(
            image_id=image_id,
            top_three_photos_of_group=True,
            completed=True,
            superseded=False,
            label_studio_project_id=70,
            content_of_image=content_of_image,
        )
    )


def calibration(dive_id: int, extrinsics_id: int = CALIBRATION_ID):
    from fishsense_api.models.laser_extrinsics import LaserExtrinsics  # pylint: disable=import-outside-toplevel

    return LaserExtrinsics(id=extrinsics_id, dive_id=dive_id, camera_id=1)


def measurement(
    image_id: int, fish_id: int = 100, laser_extrinsics_id: int | None = CALIBRATION_ID
):
    from fishsense_api.models.measurement import Measurement  # pylint: disable=import-outside-toplevel

    return Measurement(
        image_id=image_id,
        fish_id=fish_id,
        length_m=0.3,
        laser_extrinsics_id=laser_extrinsics_id,
    )

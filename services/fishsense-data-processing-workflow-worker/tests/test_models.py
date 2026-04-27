from datetime import datetime

from fishsense_data_processing_workflow_worker.models import Dive, Image


def test_dive_round_trips_through_pydantic():
    """Regression: models.py (renamed from models_tmp.py) must be importable
    and Dive must validate with realistic field values."""
    payload = {
        "id": 1,
        "name": "kelp-forest-01",
        "path": "/data/dives/kf01",
        "dive_datetime": datetime(2026, 1, 15, 10, 30, 0),
        "priority": "high",
        "flip_dive_slate": False,
        "camera_id": 7,
        "dive_slate_id": 3,
    }

    dive = Dive.model_validate(payload)

    assert dive.id == 1
    assert dive.name == "kelp-forest-01"
    assert dive.dive_datetime == datetime(2026, 1, 15, 10, 30, 0)


def test_dive_accepts_nullable_fields_as_none():
    dive = Dive.model_validate(
        {
            "id": None,
            "name": None,
            "path": "/data/dives/unnamed",
            "dive_datetime": datetime(2026, 1, 15),
            "priority": "low",
            "flip_dive_slate": None,
            "camera_id": None,
            "dive_slate_id": None,
        }
    )

    assert dive.id is None
    assert dive.flip_dive_slate is None


def test_image_round_trips_through_pydantic():
    image = Image.model_validate(
        {
            "id": 42,
            "path": "/data/images/42.orf",
            "taken_datetime": datetime(2026, 1, 15, 10, 31, 0),
            "checksum": "deadbeef",
            "is_canonical": True,
            "dive_id": 1,
            "camera_id": 7,
        }
    )

    assert image.id == 42
    assert image.is_canonical is True


def test_consumers_of_models_module_resolve():
    """Regression for the models_tmp.py → models.py rename: both modules
    that import `.models` must resolve cleanly."""
    from fishsense_data_processing_workflow_worker.activities import (  # noqa: F401
        cluster_dive_frames,
    )
    from fishsense_data_processing_workflow_worker.workflows import (  # noqa: F401
        dive_frame_clustering_workflow,
    )

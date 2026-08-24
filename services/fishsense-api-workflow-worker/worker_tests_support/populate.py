"""Fixtures shared by the four populate-activity test suites.

Laser, species, head/tail and dive-slate populate are near-identical stages,
and their tests were near-identical files: the same image and laser-label
builders and the same fake hosted-Label-Studio client, byte for byte, four
times over. `duplicate-code` flagged five pairs of them.

The fake LS client is the piece worth having in one place — it encodes how
hosted Label Studio actually behaves: imports are asynchronous, so the import
response carries no task ids, and `tasks.list` serves back a per-task presign
resolve-wrapper URL rather than the `s3://` URI that was imported. That
behaviour is exactly what the #343 dedup fix turns on, and four copies of it
are four chances to drift away from the thing being tested.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

from fishsense_api_sdk.models.image import Image
from fishsense_api_sdk.models.laser_label import LaserLabel

__all__ = [
    "image",
    "laser_label",
    "fake_label_studio_client",
    "patch_jpeg_gate",
    "laser_label_matrix",
    "images_for",
]


def image(image_id: int, checksum: str) -> Image:
    return Image(
        id=image_id,
        path=f"path/{image_id}.ORF",
        taken_datetime=datetime(2024, 8, 21, tzinfo=timezone.utc),
        checksum=checksum,
        is_canonical=True,
        dive_id=1,
        camera_id=6,
    )


def laser_label(
    image_id: int,
    *,
    completed: bool = True,
    superseded: bool = False,
    x: Optional[float] = 100.0,
    y: Optional[float] = 200.0,
) -> LaserLabel:
    return LaserLabel(
        id=image_id * 7,
        label_studio_task_id=image_id * 10,
        label_studio_project_id=43,
        x=x,
        y=y,
        label="laser",
        updated_at=None,
        superseded=superseded,
        completed=completed,
        label_studio_json={},
        image_id=image_id,
        user_id=None,
    )


def fake_label_studio_client(returned_task_ids: List[int]):
    # Fake hosted LS: import creates tasks (assigning ids from
    # `returned_task_ids` in order) and `tasks.list` serves them back. The
    # import response carries NO task_ids -- hosted LS imports asynchronously.
    ls = MagicMock()
    ls.projects = MagicMock()
    _stored: List = []
    _ids = iter(returned_task_ids)

    def _import(
        project_id, request, return_task_ids=False
    ):  # pylint: disable=unused-argument
        for task in request:
            _tid = next(_ids)
            _s3 = task["data"].get("image") or task["data"].get("img")
            _fileuri = base64.b64encode(_s3.encode()).decode()
            # hosted LS lists tasks with a per-task presign resolve-wrapper,
            # NOT the imported s3:// URL — mirror that so dedup is exercised.
            _stored.append(
                SimpleNamespace(
                    id=_tid,
                    data={"image": f"/tasks/{_tid}/resolve/?fileuri={_fileuri}"},
                )
            )
        return SimpleNamespace(import_=1)

    ls.projects.import_tasks = MagicMock(side_effect=_import)
    ls.tasks = MagicMock()
    ls.tasks.list = MagicMock(side_effect=lambda project=None: list(_stored))
    return ls


def patch_jpeg_gate(monkeypatch, module, present: bool = True):
    """Point a populate activity's object-store client at a fake.

    Every populate suite defaults this gate to "the JPEG is there" so its
    tests decide populate purely on label state; the gate's own tests override
    it with a selective fake. Takes the module rather than patching a fixed
    target because each suite patches its own activity module.
    """
    store = MagicMock()
    store.has_processed_jpeg = AsyncMock(return_value=present)
    monkeypatch.setattr(module, "open_object_store_client", lambda: store)
    return store


def laser_label_matrix() -> List[LaserLabel]:
    """Six laser labels covering every branch of the *valid laser* gate.

    Ids 1-6 are, in order: valid, incomplete, valid, superseded, null x,
    null y. The suites that cascade from a valid laser (species, head/tail)
    assert the same selection over exactly this set.
    """
    return [
        laser_label(1),
        laser_label(2, completed=False),
        laser_label(3),
        laser_label(4, superseded=True),
        laser_label(5, x=None),
        laser_label(6, y=None),
    ]


def images_for(count: int = 6) -> dict:
    """`{image_id: Image}` for ids 1..count, checksums 'a', 'b', ..."""
    return {i: image(i, chr(ord("a") + i - 1)) for i in range(1, count + 1)}

"""A label PUT must update the row it means, or refuse.

`_upsert_label` gained partial-update semantics: it writes only the fields the
body mentions. That fixed the flag-clobbering incident, but it made *which row*
gets written a separate question from *what* gets written -- and the row is
still chosen from the fully-defaulted payload, where an absent
`label_studio_project_id` is indistinguishable from an explicit null.

Two ways that goes wrong, both silent, both leaving the database worse than a
rejected request would have:

* A body that omits the project id resolves the natural key against
  `label_studio_project_id IS NULL`, misses the real row, and INSERTs a second
  one. The stray row is a sentinel: `completed` is false, so
  `dive_pipeline_status.laser_labeling_complete` reads false for that dive
  until someone finds and deletes it by hand.
* A body carrying an `id` that belongs to another image's row updates *that*
  row, and -- because `image_id` is always treated as provided -- moves it onto
  the URL's image. One image loses its label and another gains one holding the
  first image's coordinates.

Both are about targeting, so they live together here; the field-level
preservation property is in
`test_label_put_preserves_unmentioned_fields.py`.
"""

from __future__ import annotations

import pytest
from sqlmodel import select

from tests_support.db import dive, image, reprocess_label_kinds

#: Route per kind. Driven over HTTP on purpose: the targeting bug is reached
#: through what a real JSON body leaves *unset*, which a direct call to the
#: handler cannot reproduce -- building the model in Python sets every key.
_ROUTES = {
    "LaserLabel": "laser",
    "SpeciesLabel": "species",
    "HeadTailLabel": "headtail",
    "DiveSlateLabel": "dive-slate",
}

IMAGE_ID = 11
OTHER_IMAGE_ID = 12
PROJECT_ID = 73


@pytest.fixture
async def http(session):
    """The app over the in-memory session, without running the real lifespan."""
    from fastapi.testclient import TestClient

    from tests_support.app import seed_placeholder_settings

    seed_placeholder_settings()

    import fishsense_api.controllers  # noqa: F401  pylint: disable=unused-import
    from fishsense_api.database import get_async_session
    from fishsense_api.server import app

    async def _override():
        yield session

    app.dependency_overrides[get_async_session] = _override
    yield TestClient(app)
    app.dependency_overrides.pop(get_async_session, None)


async def _seed_images(session):
    session.add(dive(1))
    session.add(image(IMAGE_ID, 1))
    session.add(image(OTHER_IMAGE_ID, 1))
    await session.flush()


async def _rows(session, model, image_id: int):
    return (
        await session.exec(select(model).where(model.image_id == image_id))
    ).all()


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestOmittedProjectId:
    async def test_it_updates_the_only_row_rather_than_adding_a_null_one(
        self, http, session, model
    ):
        """The image has exactly one label, so the request is unambiguous."""
        await _seed_images(session)
        session.add(
            model(
                image_id=IMAGE_ID,
                label_studio_project_id=PROJECT_ID,
                completed=False,
                label_studio_task_id=1,
            )
        )
        await session.flush()

        response = http.put(
            f"/api/v1/labels/{_ROUTES[model.__name__]}/{IMAGE_ID}",
            json={"image_id": IMAGE_ID, "completed": True},
        )
        assert response.status_code == 201, response.text

        rows = await _rows(session, model, IMAGE_ID)
        assert len(rows) == 1, "a second, project-less row must not appear"
        assert rows[0].completed is True
        assert rows[0].label_studio_project_id == PROJECT_ID, (
            "an omitted project id must not be written as NULL"
        )

    async def test_it_refuses_when_the_image_has_several_labels(
        self, http, session, model
    ):
        """Two projects hold a label for this image, so "the" row is undefined.

        Picking one would be a coin flip over which project's label gets the
        caller's data. Refusing is the only answer that cannot corrupt.
        """
        await _seed_images(session)
        for project_id in (PROJECT_ID, PROJECT_ID + 1):
            session.add(
                model(
                    image_id=IMAGE_ID,
                    label_studio_project_id=project_id,
                    completed=False,
                )
            )
        await session.flush()

        response = http.put(
            f"/api/v1/labels/{_ROUTES[model.__name__]}/{IMAGE_ID}",
            json={"image_id": IMAGE_ID, "completed": True},
        )
        assert response.status_code == 409, response.text

        rows = await _rows(session, model, IMAGE_ID)
        assert len(rows) == 2, "a refused request must write nothing"
        assert all(row.completed is False for row in rows)

    async def test_a_first_label_is_still_created(self, http, session, model):
        """Nothing to update, so the insert path is unchanged."""
        await _seed_images(session)

        response = http.put(
            f"/api/v1/labels/{_ROUTES[model.__name__]}/{IMAGE_ID}",
            json={"image_id": IMAGE_ID, "completed": True},
        )
        assert response.status_code == 201, response.text

        rows = await _rows(session, model, IMAGE_ID)
        assert len(rows) == 1
        assert rows[0].completed is True


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestExplicitIdOwnership:
    async def test_it_refuses_an_id_belonging_to_another_image(
        self, http, session, model
    ):
        await _seed_images(session)
        victim = model(
            image_id=OTHER_IMAGE_ID,
            label_studio_project_id=PROJECT_ID,
            completed=True,
            label_studio_task_id=500,
        )
        session.add(victim)
        await session.flush()
        victim_id = victim.id

        response = http.put(
            f"/api/v1/labels/{_ROUTES[model.__name__]}/{IMAGE_ID}",
            json={
                "id": victim_id,
                "image_id": IMAGE_ID,
                "label_studio_project_id": PROJECT_ID,
                "completed": False,
            },
        )
        assert response.status_code == 409, response.text

        assert not await _rows(session, model, IMAGE_ID), (
            "the request must not have created anything either"
        )
        survivors = await _rows(session, model, OTHER_IMAGE_ID)
        assert len(survivors) == 1, "the other image must keep its label"
        assert survivors[0].id == victim_id
        assert survivors[0].completed is True, "and keep its values"

    async def test_it_accepts_an_id_belonging_to_this_image(
        self, http, session, model
    ):
        """The guard is about ownership, not about forbidding explicit ids."""
        await _seed_images(session)
        own = model(
            image_id=IMAGE_ID,
            label_studio_project_id=PROJECT_ID,
            completed=False,
        )
        session.add(own)
        await session.flush()

        response = http.put(
            f"/api/v1/labels/{_ROUTES[model.__name__]}/{IMAGE_ID}",
            json={
                "id": own.id,
                "image_id": IMAGE_ID,
                "label_studio_project_id": PROJECT_ID,
                "completed": True,
            },
        )
        assert response.status_code == 201, response.text

        rows = await _rows(session, model, IMAGE_ID)
        assert len(rows) == 1
        assert rows[0].completed is True

"""A PUT must not clear fields the caller never mentioned.

The four `put_*_label` handlers rebuilt a full SQLModel from the request body
and `session.merge`d the whole thing. Anything absent from the body took its
model default -- so a writer that constructs a label with the twelve fields it
cares about silently wiped the ones it did not.

Prod, 2026-09-07: `populate_laser_label_studio_project_activity._record` builds
a fresh `LaserLabel` with thirteen explicit kwargs and no `needs_reprocess`,
and its `_select_unlabeled_images` selects images with no *completed* label --
which is exactly the set a reprocess flag marks. Populate runs hourly, so a
laser reprocess flag had a useful life of under an hour: dive 442's 259 flags
were set at 22:05 and gone before the render they requested had started.

The fix is at the API rather than in each caller because "remember to restate
every field you do not want cleared" is a rule every future writer would have
to keep.

**It does not cover read-modify-write callers, and cannot.** The hourly syncs
fetch a label, mutate two fields and PUT the whole model back
(`sync_laser_labels_for_label_studio_project_activity`), so every field is
genuinely present in their body -- including `needs_reprocess: false` as it
read it. A flag raised between that read and that write is still lost. Closing
that needs the flag to stop being writable through the label PUT at all, which
is a deliberate API change and not this fix; see the note in `_upsert_label`.

`LaserPrediction` shares the *mechanism* but not the verdict: its gate fields
are cleared on re-prediction on purpose, and that clobber is load-bearing --
see the note in `_upsert_label` before propagating anything here to
`_prediction_upsert.py`.
"""

from __future__ import annotations

import pytest
from sqlmodel import select

from tests_support.db import dive, image, reprocess_label_kinds

# Suite-local: only these tests drive the PUT handlers, so this is a mapping
# this file owns rather than another copy of the shared kinds list.
def _handler_for(model):
    from fishsense_api.controllers import label_controller as lc

    return {
        "LaserLabel": lc.put_laser_label,
        "SpeciesLabel": lc.put_species_label,
        "HeadTailLabel": lc.put_headtail_label,
        "DiveSlateLabel": lc.put_dive_slate_label,
    }[model.__name__]


#: Route per kind, for the over-HTTP test below.
_ROUTES = {
    "LaserLabel": "laser",
    "SpeciesLabel": "species",
    "HeadTailLabel": "headtail",
    "DiveSlateLabel": "dive-slate",
}

IMAGE_ID = 11
PROJECT_ID = 7


async def _seed_image(session, image_id: int = IMAGE_ID):
    session.add(dive(1))
    session.add(image(image_id, 1))
    await session.flush()


@pytest.mark.parametrize("model", reprocess_label_kinds())
class TestUnmentionedFieldsSurvive:
    async def test_a_put_that_omits_needs_reprocess_does_not_clear_it(
        self, session, model
    ):
        """The prod case: populate rewrites the row and the flag disappears."""
        await _seed_image(session)
        session.add(
            model(
                image_id=IMAGE_ID,
                label_studio_project_id=PROJECT_ID,
                completed=False,
                needs_reprocess=True,
            )
        )
        await session.flush()

        # what populate sends: explicit kwargs, no `needs_reprocess`
        payload = model(
            image_id=IMAGE_ID,
            label_studio_project_id=PROJECT_ID,
            label_studio_task_id=99,
            completed=False,
        )
        await _handler_for(model)(IMAGE_ID, payload, session=session)
        await session.flush()

        rows = (
            await session.exec(select(model).where(model.image_id == IMAGE_ID))
        ).all()
        assert len(rows) == 1, "natural-key upsert must not append a row"
        assert rows[0].needs_reprocess is True, (
            "a field the caller never mentioned must survive the write"
        )
        assert rows[0].label_studio_task_id == 99, "provided fields still apply"

    async def test_a_put_that_explicitly_sends_false_does_clear_it(
        self, session, model
    ):
        """Preserving the unmentioned must not make the field unwritable."""
        await _seed_image(session)
        session.add(
            model(
                image_id=IMAGE_ID,
                label_studio_project_id=PROJECT_ID,
                needs_reprocess=True,
            )
        )
        await session.flush()

        payload = model(
            image_id=IMAGE_ID,
            label_studio_project_id=PROJECT_ID,
            needs_reprocess=False,
        )
        await _handler_for(model)(IMAGE_ID, payload, session=session)
        await session.flush()

        rows = (
            await session.exec(select(model).where(model.image_id == IMAGE_ID))
        ).all()
        assert rows[0].needs_reprocess is False

    async def test_a_new_row_is_still_created(self, session, model):
        await _seed_image(session)
        payload = model(
            image_id=IMAGE_ID, label_studio_project_id=PROJECT_ID, completed=True
        )
        await _handler_for(model)(IMAGE_ID, payload, session=session)
        await session.flush()

        rows = (
            await session.exec(select(model).where(model.image_id == IMAGE_ID))
        ).all()
        assert len(rows) == 1
        assert rows[0].completed is True


@pytest.fixture
async def http(session):
    """The app over the in-memory session, without running the real lifespan.

    Entering `TestClient`'s context manager would run `lifespan`, which calls
    `create_all` against Postgres and then `run_alembic_upgrade`.
    """
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


@pytest.mark.parametrize("model", reprocess_label_kinds())
async def test_the_flag_survives_a_real_request_body(http, session, model):
    """The same property, but crossing HTTP.

    The whole fix rests on `payload.model_fields_set` holding exactly the keys
    the caller sent -- and that is a property of how FastAPI validates a
    request body into a `table=True` SQLModel, which a direct call to the
    handler never exercises: those tests hand the handler a model they built
    themselves, so they set `model_fields_set` rather than observing it.

    Without this test a sqlmodel or pydantic bump, or a switch to
    `Body(embed=True)` or a request DTO, could restore the prod defect with the
    entire suite green.
    """
    await _seed_image(session)
    session.add(
        model(
            image_id=IMAGE_ID,
            label_studio_project_id=PROJECT_ID,
            completed=False,
            needs_reprocess=True,
        )
    )
    await session.flush()

    response = http.put(
        f"/api/v1/labels/{_ROUTES[model.__name__]}/{IMAGE_ID}",
        json={
            "image_id": IMAGE_ID,
            "label_studio_project_id": PROJECT_ID,
            "label_studio_task_id": 99,
            "completed": False,
        },
    )
    assert response.status_code == 201, response.text

    rows = (
        await session.exec(select(model).where(model.image_id == IMAGE_ID))
    ).all()
    assert len(rows) == 1
    assert rows[0].needs_reprocess is True, (
        "a JSON body that omits the flag must not clear it"
    )
    assert rows[0].label_studio_task_id == 99

"""What `schedule_workflows` actually registers.

The hourly parents are deliberately staggered across the hour so their
selectors don't all hit fishsense-api at the same moment (CLAUDE.md,
"Cross-worker orchestration pattern"). That stagger only exists as a
pile of `offset=timedelta(minutes=N)` literals, so two schedules landing
on the same slot — or a parent silently never being registered — is a
mistake nothing else catches.

These are the first tests over the registration itself; the seven
pre-existing schedules had none. `ensure_schedule` is patched out, so
nothing here talks to Temporal.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from temporalio.client import ScheduleOverlapPolicy

from fishsense_api_workflow_worker import worker as sut

# The hourly parents that pick a dive off a cohort selector. These are
# the ones the stagger exists for; the label-studio sync schedules are
# excluded deliberately (they select no dives and share offset 0).
_DIVE_SELECTING_PARENT_SCHEDULE_IDS = (
    "preprocess-laser-images-workflow-schedule",
    "cluster-dive-frames-workflow-schedule",
    "preprocess-species-images-workflow-schedule",
    "populate-species-labels-workflow-schedule",
    "preprocess-headtail-images-workflow-schedule",
    "preprocess-slate-images-workflow-schedule",
    "perform-laser-calibration-workflow-schedule",
    "measure-fish-workflow-schedule",
)


@pytest.fixture
async def registered(monkeypatch):
    """Run `schedule_workflows` with Temporal stubbed; return
    {schedule_id: Schedule}."""
    captured: dict = {}

    async def _fake_ensure_schedule(_client, *, schedule_id, schedule):
        captured[schedule_id] = schedule

    monkeypatch.setattr(sut, "ensure_schedule", _fake_ensure_schedule)
    await sut.schedule_workflows(MagicMock())
    return captured


def _offset(schedule) -> timedelta | None:
    return schedule.spec.intervals[0].offset


def _every(schedule) -> timedelta:
    return schedule.spec.intervals[0].every


async def test_measure_fish_is_scheduled_hourly_at_40(registered):
    """Stage 14 went from operator-only to scheduled on 2026-07-17."""
    schedule = registered["measure-fish-workflow-schedule"]

    assert _every(schedule) == timedelta(hours=1)
    assert _offset(schedule) == timedelta(minutes=40)


async def test_species_populate_is_scheduled_hourly_at_20(registered):
    """The decoupled species-populate parent fires at +20, just after the
    +15 species-preprocess writes JPEGs, with SKIP overlap like the other
    dive-selecting parents."""
    schedule = registered["populate-species-labels-workflow-schedule"]

    assert _every(schedule) == timedelta(hours=1)
    assert _offset(schedule) == timedelta(minutes=20)
    assert schedule.policy.overlap is ScheduleOverlapPolicy.SKIP


async def test_measure_fish_skips_when_a_run_is_still_in_flight(registered):
    """SKIP, not ALLOW_ALL: two selectors racing would both pick the same
    dive. Every dive-selecting parent shares this."""
    schedule = registered["measure-fish-workflow-schedule"]

    assert schedule.policy.overlap is ScheduleOverlapPolicy.SKIP


async def test_measure_fish_run_timeout_outlives_its_child(registered):
    """The child's `execution_timeout` is 1h; the parent must outlast it
    plus the selector and data-worker scale-up activities."""
    schedule = registered["measure-fish-workflow-schedule"]

    assert schedule.action.run_timeout > timedelta(hours=1)


async def test_dive_selecting_parents_do_not_share_a_slot(registered):
    """The stagger is the whole point — two selectors firing at the same
    instant would both `dives.get()` at once, and (SKIP overlap only
    guards a schedule against *itself*) could pick the same dive.

    Scoped to the dive-selecting parents on purpose. The label-studio
    sync schedules deliberately share offset 0: they don't select dives,
    and they run `overlap=ALLOW_ALL`.
    """
    slots: dict[timedelta, list[str]] = {}
    for schedule_id in _DIVE_SELECTING_PARENT_SCHEDULE_IDS:
        schedule = registered[schedule_id]
        slots.setdefault(_offset(schedule) or timedelta(0), []).append(schedule_id)

    collisions = {
        offset: ids for offset, ids in slots.items() if len(ids) > 1
    }
    assert not collisions, f"parents share a slot: {collisions}"


async def test_measure_fish_lands_before_the_scale_down_sweeper(registered):
    """The +55 sweeper scales the NRP data-worker to zero. Stage 14 must
    fire early enough to have scaled it *up* and be visibly running by
    then, or the sweeper could pull the floor out mid-measure."""
    measure = _offset(registered["measure-fish-workflow-schedule"])
    sweeper = _offset(registered["scale-down-idle-data-worker-workflow-schedule"])

    assert measure < sweeper


async def test_every_dive_selecting_parent_is_registered(registered):
    """Guards against a parent being added to the worker but never
    scheduled — it would simply never run."""
    for schedule_id in _DIVE_SELECTING_PARENT_SCHEDULE_IDS:
        assert schedule_id in registered


async def test_labeling_config_reconcile_is_scheduled_hourly_at_25(registered):
    """Slot +25 — the gap between the +20 species populate and the +30
    headtail preprocess. It selects no dive, so it isn't part of the
    dive-selecting stagger, but it must still not collide with a parent."""
    schedule = registered["reconcile-labeling-configs-workflow-schedule"]

    assert _every(schedule) == timedelta(hours=1)
    assert _offset(schedule) == timedelta(minutes=25)
    assert schedule.spec.intervals[0].offset not in {
        _offset(registered[sid]) for sid in _DIVE_SELECTING_PARENT_SCHEDULE_IDS
    }


async def test_labeling_config_reconcile_skips_when_still_in_flight(registered):
    """A slow workspace walk must not stack — the next hour re-converges."""
    schedule = registered["reconcile-labeling-configs-workflow-schedule"]

    assert schedule.policy.overlap == ScheduleOverlapPolicy.SKIP

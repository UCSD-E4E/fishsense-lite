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
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.client import ScheduleOverlapPolicy
from temporalio.service import RPCError, RPCStatusCode

from fishsense_api_workflow_worker import worker as sut

# The hourly parents that pick a dive off a cohort selector. These are
# the ones the stagger exists for; the label-studio sync schedules are
# excluded deliberately (they select no dives and share offset 0).
_DIVE_SELECTING_PARENT_SCHEDULE_IDS = (
    "preprocess-laser-images-workflow-schedule",
    "predict-laser-images-workflow-schedule",
    "populate-laser-labels-workflow-schedule",
    "cluster-dive-frames-workflow-schedule",
    "preprocess-species-images-workflow-schedule",
    "populate-species-labels-workflow-schedule",
    "preprocess-headtail-images-workflow-schedule",
    "preprocess-slate-images-workflow-schedule",
    "perform-laser-calibration-workflow-schedule",
    "measure-fish-workflow-schedule",
    "compute-laser-depths-workflow-schedule",
)


@pytest.fixture
async def registered(monkeypatch):
    """Run `schedule_workflows` with Temporal stubbed; return
    {schedule_id: Schedule}."""
    captured: dict = {}

    async def _fake_ensure_schedule(_client, *, schedule_id, schedule):
        captured[schedule_id] = schedule

    async def _fake_retire_schedule(_client, _schedule_id):
        return None

    monkeypatch.setattr(sut, "ensure_schedule", _fake_ensure_schedule)
    monkeypatch.setattr(sut, "retire_schedule", _fake_retire_schedule)
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


async def test_laser_depth_is_scheduled_hourly_at_35(registered):
    """Slot +35 — vacated when the slate detector was shut down, and the only
    gap left between the headtail parent (+30) and stage 14 (+40)."""
    schedule = registered["compute-laser-depths-workflow-schedule"]

    assert _every(schedule) == timedelta(hours=1)
    assert _offset(schedule) == timedelta(minutes=35)


async def test_laser_depth_skips_when_still_in_flight(registered):
    """SKIP overlap, like every dive-selecting parent: two runs racing past
    the same selector would both claim the same dive."""
    schedule = registered["compute-laser-depths-workflow-schedule"]

    assert schedule.policy.overlap == ScheduleOverlapPolicy.SKIP


async def test_laser_predict_is_scheduled_hourly_at_10(registered):
    schedule = registered["predict-laser-images-workflow-schedule"]
    assert _every(schedule) == timedelta(hours=1)
    assert _offset(schedule) == timedelta(minutes=10)


async def test_slate_predict_is_not_scheduled(registered):
    """Deliberately unscheduled: the slate estimator's ECC gate was calibrated
    only on clear-water reef and produces high-ECC false fits on out-of-
    distribution frames (e.g. pool calibration dives), so automated seeding is
    off until the detector is validated on those conditions. The parent stays
    registered for on-demand use; only the schedule is withheld."""
    assert "predict-slate-images-workflow-schedule" not in registered


async def test_slate_predict_schedule_is_actively_retired(monkeypatch):
    """Not-created isn't enough: a prior deploy may already have created the
    schedule, so startup must actively delete it (idempotently) or a stale
    schedule keeps firing. Pins that `schedule_workflows` retires it."""
    retired: list[str] = []

    async def _fake_ensure_schedule(_client, *, schedule_id, schedule):
        # pylint: disable=unused-argument
        return None

    async def _fake_retire_schedule(_client, schedule_id):
        retired.append(schedule_id)

    monkeypatch.setattr(sut, "ensure_schedule", _fake_ensure_schedule)
    monkeypatch.setattr(sut, "retire_schedule", _fake_retire_schedule)
    await sut.schedule_workflows(MagicMock())

    assert "predict-slate-images-workflow-schedule" in retired


def _handle_raising(exc):
    handle = MagicMock()
    handle.delete = AsyncMock(side_effect=exc)
    client = MagicMock()
    client.get_schedule_handle = MagicMock(return_value=handle)
    return client, handle


async def test_retire_schedule_deletes_when_present():
    handle = MagicMock()
    handle.delete = AsyncMock()
    client = MagicMock()
    client.get_schedule_handle = MagicMock(return_value=handle)

    await sut.retire_schedule(client, "some-schedule")

    client.get_schedule_handle.assert_called_once_with("some-schedule")
    handle.delete.assert_awaited_once()


async def test_retire_schedule_is_idempotent_on_not_found():
    client, _ = _handle_raising(RPCError("missing", RPCStatusCode.NOT_FOUND, b""))
    # Absent schedule is the desired state -> no raise.
    await sut.retire_schedule(client, "some-schedule")


async def test_retire_schedule_reraises_other_rpc_errors():
    client, _ = _handle_raising(RPCError("boom", RPCStatusCode.INTERNAL, b""))
    with pytest.raises(RPCError):
        await sut.retire_schedule(client, "some-schedule")


async def test_laser_populate_is_scheduled_hourly_at_12(registered):
    """Runs after the +10 predict parent (predictions must exist first)."""
    schedule = registered["populate-laser-labels-workflow-schedule"]
    assert _every(schedule) == timedelta(hours=1)
    assert _offset(schedule) == timedelta(minutes=12)


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

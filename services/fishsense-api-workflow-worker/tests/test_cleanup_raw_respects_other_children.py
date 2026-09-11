# pylint: disable=unused-argument,protected-access
"""Raw scratch is shared per DIVE, so cleanup must respect other stages.

The staged objects are keyed `raw/{checksum}.ORF` -- per dive, not per stage --
which is deliberate: a dive in several cohorts stages once and the later stages
report `skipped_already_present`. The corollary is that *deleting* is a
cross-stage act, and the child-id sentinel cannot see it: the ids differ
(`preprocess-laser-442` vs `preprocess-species-442`), so no
`WorkflowAlreadyStartedError` is ever raised.

The sequence that loses a render:

    :00  stage 0.1 dispatches preprocess-laser-442, which runs ~1h
    :15  stage 2 dispatches preprocess-species-442, which finishes quickly
    :17  the species parent calls cleanup_raw(442) and deletes every raw
         object -- including the ones the laser child is still reading
    ...  the laser child dies with NoSuchKey, having half-rendered

So the delete is gated inside the *activity* rather than at a call site: it is
one place, it protects every caller including future ones, and it adds no
workflow command, so no in-flight parent's replay contract changes.

The last stage out does the cleanup. If it fails instead, the scratch leaks
until the next firing re-stages -- which is cheap (`skipped_already_present`)
and is the pre-existing failure direction anyway.
"""

from __future__ import annotations


from fishsense_api_workflow_worker.activities.cleanup_raw_bytes_for_dive_activity import (  # noqa: E501  pylint: disable=line-too-long
    raw_scratch_reader_ids,
    build_scratch_in_use_query,
)


class TestReaderIds:
    def test_covers_every_child_that_reads_raw_scratch(self):
        """Preprocess, predict and both checkerboard children download
        `raw/{checksum}.ORF`. Missing one means cleanup can delete under it.

        The two checkerboard ids were absent until 2026-09-11 — the calibration
        child had been missing since it shipped on 2026-09-07 — so a preprocess
        cleanup could evict a calibration dive's scratch mid-fit.
        """
        assert raw_scratch_reader_ids(442) == [
            "preprocess-laser-442",
            "preprocess-species-442",
            "preprocess-headtail-442",
            "preprocess-slate-442",
            "predict-laser-442",
            "predict-slate-442",
            "perform-checkerboard-calibration-442",
            "verify-checkerboard-lattice-442",
        ]

    def test_every_dispatched_child_id_shape_is_covered(self):
        """A tripwire against the next stage forgetting.

        The ids here are the `child_id=` arguments the parent workflows pass to
        `_dispatch.dispatch_child`. Anything that stages raw bytes and then
        dispatches a child must appear, so this asserts the shapes rather than
        trusting a reviewer to notice the omission.
        """
        ids = set(raw_scratch_reader_ids(1))
        for prefix in (
            "preprocess-laser",
            "preprocess-species",
            "preprocess-headtail",
            "preprocess-slate",
            "predict-laser",
            "predict-slate",
            "perform-checkerboard-calibration",
            "verify-checkerboard-lattice",
        ):
            assert f"{prefix}-1" in ids


class TestQuery:
    def test_asks_only_for_running_children_of_this_dive(self):
        q = build_scratch_in_use_query(442)
        assert 'ExecutionStatus = "Running"' in q
        assert "preprocess-laser-442" in q
        assert "predict-slate-442" in q

    def test_does_not_match_a_different_dive(self):
        """`WorkflowId IN (...)` and not a prefix match: dive 44 must not keep
        dive 442's scratch alive, and 4420's must not keep 442's."""
        q = build_scratch_in_use_query(44)
        assert "preprocess-laser-44'" in q or '"preprocess-laser-44"' in q
        assert "preprocess-laser-442" not in q

"""The fishsense-api <-> api-worker ingest contract.

These DTOs cross a package boundary that has no import edge in either
direction, so a field renamed on one side and not the other fails at runtime in
production rather than at import in CI. The defaults are the part worth
pinning: several of them encode decisions that are invisible in the type.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fishsense_shared import (
    DuplicateOverlap,
    IngestDiveRequest,
    IngestPreflight,
    IngestProgress,
    IngestReport,
    PreflightImage,
    RejectedImage,
    SubfolderReport,
)
from fishsense_shared.ingest_contracts import (
    ChecksumMismatch,
    DiveVerificationSummary,
    VerifyAllDivesProgress,
    VerifyAllDivesReport,
)


def test_a_request_needs_only_a_dive_path():
    """Everything else has a defensible default; the folder is the one thing
    only the operator knows."""
    request = IngestDiveRequest(dive_path="2024 REEF/082124_FSL06")

    assert request.dive_path == "2024 REEF/082124_FSL06"
    assert request.dive_name is None


def test_priority_defaults_to_high():
    """A dive ingested at LOW is invisible to every cohort selector, so it
    would sit there looking successful and never process."""
    assert IngestDiveRequest(dive_path="d").priority == "HIGH"


def test_calibration_intent_defaults_to_neither():
    """Deliberately not defaulted to `self_calibrates=True`.

    A fish-only dive with no slate frames can never self-calibrate, and that is
    not detectable from the files — so preflight requires the operator to state
    intent. Defaulting either way would silently pick for them.
    """
    request = IngestDiveRequest(dive_path="d")

    assert request.calibration_dive_id is None
    assert request.self_calibrates is False


def test_writing_nothing_is_opt_in():
    """`dry_run` defaults False: the common case is an operator who means to
    ingest. Preflight still runs either way, so a fault is caught before
    anything is written regardless."""
    request = IngestDiveRequest(dive_path="d", self_calibrates=True)

    assert request.dry_run is False


def test_the_request_carries_no_verify_existing_flag():
    """Removed, not renamed.

    It was declared on this model and honoured by no code, so an operator
    setting it got a normal ingest and no warning — worse than the flag not
    existing. Re-hashing existing rows is `VerifyDiveChecksumsWorkflow`, which
    does it read-only and without a second copy of the checksum convention to
    keep in step.

    Pydantic ignores unknown keys, so a stale caller passing it is silently
    accepted rather than failing loudly. That is the right behaviour for
    replaying an in-flight workflow's payload, and the reason this test asserts
    the field is genuinely gone rather than trusting a rename to surface.
    """
    request = IngestDiveRequest(dive_path="d", self_calibrates=True)

    assert "verify_existing" not in request.model_dump()

    stale = IngestDiveRequest(
        dive_path="d", self_calibrates=True, verify_existing=True
    )

    assert not hasattr(stale, "verify_existing")
    assert "verify_existing" not in stale.model_dump()


def test_a_preflight_image_may_have_no_timestamp():
    """None means "reject this frame", never "use a default" — stage-1
    clustering is pure timestamp maths."""
    image = PreflightImage(path="d/P1.ORF", size=15_232_982)

    assert image.taken_datetime is None
    assert image.exif_offset is None


def test_preflight_image_records_the_offset_without_applying_it():
    """The camera writes local time plus an offset; the existing ~111k rows
    store the local value stamped UTC. Ingest reproduces that, but keeps the
    offset visible so the divergence is reported rather than lost."""
    image = PreflightImage(
        path="d/P1.ORF",
        size=1,
        taken_datetime=datetime(2025, 3, 6, 17, 0, 15, tzinfo=timezone.utc),
        exif_offset="-08:00",
    )

    assert image.exif_offset == "-08:00"
    assert image.taken_datetime.hour == 17          # not shifted by the offset


def test_a_fresh_preflight_reports_no_problems():
    preflight = IngestPreflight(dive_path="d")

    assert not preflight.errors
    assert not preflight.warnings
    assert not preflight.images
    assert not preflight.subfolders


def test_subfolders_are_reported_as_separate_dives():
    """The Olympus rollover case. Reported, never ingested — recursing would
    merge dives that are distinct rows in prod."""
    report = SubfolderReport(path="d/101923_Alligator1_FSL06", orf_count=47)

    assert report.orf_count == 47


def test_duplicate_overlap_is_a_containment_ratio():
    """A set operation over checksums, so it degrades to a partial overlap
    instead of the legacy digest's all-or-nothing answer."""
    overlap = DuplicateOverlap(dive_id=64, shared_images=48, containment=48 / 55)

    assert overlap.containment == pytest.approx(0.8727, abs=1e-4)


def test_a_report_is_uncommitted_until_proven_otherwise():
    """`committed` is the commit flag: priority only flips to HIGH when every
    listed frame landed. A partially-ingested dive must never enter the
    pipeline, so the default has to be False."""
    report = IngestReport(dive_path="d")

    assert report.committed is False
    assert report.dive_id is None
    assert not report.rejected


def test_a_rejection_carries_its_reason():
    rejected = RejectedImage(path="d/P1.ORF", reason="no EXIF DateTime")

    assert rejected.reason == "no EXIF DateTime"


def test_progress_starts_empty_so_the_portal_can_poll_immediately():
    progress = IngestProgress()

    assert progress.state == "starting"
    assert (progress.total, progress.scanned, progress.registered) == (0, 0, 0)


def test_the_contract_round_trips_through_json():
    """Temporal serializes these across the process boundary, and the portal
    reads them over HTTP. A field that cannot round-trip fails in production,
    not here."""
    report = IngestReport(
        dive_path="d",
        dive_id=7,
        total=2,
        registered=2,
        dive_datetime=datetime(2024, 8, 21, 8, 56, 51, tzinfo=timezone.utc),
        committed=True,
        rejected=[RejectedImage(path="x", reason="y")],
        duplicate_overlap=[
            DuplicateOverlap(dive_id=64, shared_images=1, containment=0.5)
        ],
    )

    restored = IngestReport.model_validate_json(report.model_dump_json())

    assert restored == report


# ── the migration audit ───────────────────────────────────────────────


def test_a_dive_with_nothing_wrong_is_clean():

    summary = DiveVerificationSummary(
        dive_id=412, checked=5, total_in_dive=55, checksum_matched=5
    )

    assert summary.is_clean is True


@pytest.mark.parametrize(
    "field",
    [
        "mismatches",
        "timestamp_mismatches",
        "missing_on_nas",
        "no_stored_checksum",
    ],
)
def test_any_kind_of_finding_makes_a_dive_not_clean(field):
    """All four are findings. They are separate fields because they mean
    different things — a wrong checksum breaks duplicate detection, a wrong
    timestamp breaks stage-1 clustering — but any of them means this dive did
    not come through the migration the way we think it did."""

    summary = DiveVerificationSummary(
        dive_id=412,
        checked=5,
        checksum_matched=4,
        **{field: [ChecksumMismatch(path="a/b.ORF")]},
    )

    assert summary.is_clean is False


def test_a_dive_that_errored_is_not_clean_even_with_no_findings():
    """The distinction the whole audit rests on. An unreachable dive has no
    findings *because it was never read* — letting that read as clean would
    quietly shrink the corpus the audit claims to cover."""

    summary = DiveVerificationSummary(dive_id=412, error="NAS unreachable")

    assert summary.is_clean is False


def test_the_sweep_report_singles_out_the_dives_with_findings():
    """Clean dives stay in `dives` — absence of a finding is the result being
    sought, so dropping them would make "verified, fine" indistinguishable from
    "never reached"."""

    report = VerifyAllDivesReport(
        dives_requested=3,
        dives_verified=3,
        dives=[
            DiveVerificationSummary(dive_id=11, checked=5, checksum_matched=5),
            DiveVerificationSummary(
                dive_id=22,
                checked=5,
                checksum_matched=4,
                mismatches=[ChecksumMismatch(path="a/b.ORF")],
            ),
            DiveVerificationSummary(dive_id=33, error="NAS unreachable"),
        ],
    )

    assert [d.dive_id for d in report.dives_with_findings] == [22, 33]
    assert len(report.dives) == 3


def test_the_audit_contract_round_trips_through_json():
    """Temporal serializes the sweep's return value, and a ~479-dive report is
    the largest payload in this contract."""

    report = VerifyAllDivesReport(
        dives_requested=1,
        dives_verified=1,
        images_checked=5,
        checksum_matched=4,
        dives=[
            DiveVerificationSummary(
                dive_id=22,
                checked=5,
                checksum_matched=4,
                mismatches=[
                    ChecksumMismatch(
                        image_id=9,
                        path="a/b.ORF",
                        stored="0" * 32,
                        computed="1" * 32,
                    )
                ],
            )
        ],
    )
    progress = VerifyAllDivesProgress(state="verifying", total_dives=479)

    assert VerifyAllDivesReport.model_validate_json(report.model_dump_json()) == report
    assert (
        VerifyAllDivesProgress.model_validate_json(progress.model_dump_json())
        == progress
    )

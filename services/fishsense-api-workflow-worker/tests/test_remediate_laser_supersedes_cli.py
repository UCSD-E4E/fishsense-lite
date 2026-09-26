"""The operator CLI for laser-supersede remediation.

What it must guarantee, because it is the thing a person runs against prod:

* no arguments means a dry run;
* apply takes EVERYTHING from the reviewed report — its dives, its exclusions
  and its digest — so what is written is what was reviewed, and nothing typed
  on the command line at apply time can widen it;
* apply cannot be reached without naming a report.
"""

from __future__ import annotations

import json

import pytest

from fishsense_api_workflow_worker import remediate_laser_supersedes as cli


def test_no_flags_is_a_dry_run_over_the_named_dives():
    args = cli.parse_args(["dry-run", "--dives", "7,9", "--out", "r.json"])

    request = cli.build_request(args, all_dive_ids=[1, 2, 3])

    assert request.apply is False
    assert request.dive_ids == [7, 9]
    assert request.expected_plan_sha256 is None


def test_without_dives_it_plans_every_dive():
    args = cli.parse_args(["dry-run", "--out", "r.json"])

    assert cli.build_request(args, all_dive_ids=[3, 1, 2]).dive_ids == [1, 2, 3]


def test_exclusions_come_from_a_file(tmp_path):
    exclusions = tmp_path / "exclusions.json"
    exclusions.write_text(json.dumps({"dive_ids": [77], "label_ids": [5, 6]}))
    args = cli.parse_args(
        ["dry-run", "--out", "r.json", "--exclusions", str(exclusions)]
    )

    request = cli.build_request(args, all_dive_ids=[77, 78])

    assert request.excluded_dive_ids == [77]
    assert request.excluded_label_ids == [5, 6]


def test_apply_takes_everything_from_the_reviewed_report(tmp_path):
    report = {
        "mode": "dry_run",
        "plan_sha256": "f" * 64,
        "excluded_dive_ids": [77],
        "excluded_label_ids": [5],
        "dives": [{"dive_id": 7}, {"dive_id": 77}],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    args = cli.parse_args(["apply", "--report", str(path)])

    request = cli.build_request(args, all_dive_ids=[1, 7, 77, 99])

    assert request.apply is True
    assert request.expected_plan_sha256 == "f" * 64
    assert request.dive_ids == [7, 77]
    assert request.excluded_dive_ids == [77]
    assert request.excluded_label_ids == [5]


def test_apply_needs_a_report():
    with pytest.raises(SystemExit):
        cli.parse_args(["apply"])


def test_apply_refuses_a_report_that_is_not_a_dry_run(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"mode": "apply", "plan_sha256": "f" * 64, "dives": []}))
    args = cli.parse_args(["apply", "--report", str(path)])

    with pytest.raises(SystemExit):
        cli.build_request(args, all_dive_ids=[])


def test_the_report_is_written_as_json(tmp_path):
    out = tmp_path / "r.json"
    cli.write_report({"mode": "dry_run", "dives": []}, str(out))

    assert json.loads(out.read_text())["mode"] == "dry_run"

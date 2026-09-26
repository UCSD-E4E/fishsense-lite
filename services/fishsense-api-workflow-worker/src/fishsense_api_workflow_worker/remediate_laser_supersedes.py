"""Operator CLI: revive laser labels the eroding validator superseded.

Run inside the api-worker container (it has the Temporal client and settings):

    python -m fishsense_api_workflow_worker.remediate_laser_supersedes \\
        dry-run --out /tmp/report.json [--dives 7,9] [--exclusions excl.json]

    python -m fishsense_api_workflow_worker.remediate_laser_supersedes \\
        apply --report /tmp/report.json

`dry-run` writes nothing and produces the report. `apply` takes its dives,
exclusions and plan digest from the reviewed report and nothing else; the
data-worker recomputes the plan and refuses unless the digest still matches.
`excl.json` is `{"dive_ids": [...], "label_ids": [...]}`.

See `RemediateLaserSupersedesParentWorkflow` and CLAUDE.md's laser-validation
section for what a revival does downstream.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from typing import Sequence

from fishsense_shared.laser_remediation import (
    PARENT_WORKFLOW,
    RemediateLaserSupersedesInput,
)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """`dry-run` (the default action) or `apply --report`."""
    parser = argparse.ArgumentParser(prog=__package__ + ".remediate_laser_supersedes")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run", help="plan every dive; write nothing")
    dry.add_argument("--out", required=True, help="where to write the report")
    dry.add_argument("--dives", help="comma-separated dive ids (default: all)")
    dry.add_argument("--exclusions", help='JSON {"dive_ids": [], "label_ids": []}')
    apply = sub.add_parser("apply", help="apply a reviewed dry-run report")
    apply.add_argument("--report", required=True, help="the reviewed report")
    apply.add_argument("--out", help="where to write the apply report")
    return parser.parse_args(argv)


def build_request(
    args: argparse.Namespace, all_dive_ids: Sequence[int]
) -> RemediateLaserSupersedesInput:
    """The workflow input for these arguments."""
    if args.command == "apply":
        with open(args.report, encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("mode") != "dry_run":
            sys.exit(f"{args.report} is not a dry-run report; refusing to apply it")
        return RemediateLaserSupersedesInput(
            dive_ids=sorted(row["dive_id"] for row in report["dives"]),
            excluded_dive_ids=list(report.get("excluded_dive_ids", [])),
            excluded_label_ids=list(report.get("excluded_label_ids", [])),
            apply=True,
            expected_plan_sha256=report["plan_sha256"],
        )

    exclusions = {"dive_ids": [], "label_ids": []}
    if args.exclusions:
        with open(args.exclusions, encoding="utf-8") as handle:
            exclusions.update(json.load(handle))
    dive_ids = (
        [int(d) for d in args.dives.split(",") if d.strip()]
        if args.dives
        else list(all_dive_ids)
    )
    return RemediateLaserSupersedesInput(
        dive_ids=sorted(dive_ids),
        excluded_dive_ids=[int(d) for d in exclusions["dive_ids"]],
        excluded_label_ids=[int(i) for i in exclusions["label_ids"]],
    )


def write_report(report: dict, path: str) -> None:
    """The report as indented JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


async def _all_dive_ids() -> list[int]:
    # pylint: disable=import-outside-toplevel
    #   Function-local so importing this module for its pure helpers does not
    #   trigger Dynaconf's eager validation (see CLAUDE.md).
    from fishsense_api_workflow_worker.activities.utils import get_fs_client

    async with get_fs_client() as fs:
        return [dive.id for dive in await fs.dives.get() or []]


async def _run(args: argparse.Namespace) -> None:
    # pylint: disable=import-outside-toplevel
    from temporalio.client import Client

    from fishsense_api_workflow_worker import worker
    from fishsense_api_workflow_worker.config import settings

    all_ids = (
        await _all_dive_ids() if args.command == "dry-run" and not args.dives else []
    )
    request = build_request(args, all_ids)
    client = await Client.connect(
        f"{settings.temporal.host}:{settings.temporal.port}",
        tls=worker.build_tls_config(settings.temporal),
        namespace=worker.temporal_namespace(settings.temporal),
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = await client.execute_workflow(
        PARENT_WORKFLOW,
        request,
        id=f"remediate-laser-supersedes-{args.command}-{stamp}",
        task_queue=worker.TASK_QUEUE_NAME,
    )
    out = args.out or f"{args.report}.applied.json"
    write_report(report, out)
    totals = report["totals"]
    print(
        f"{report['mode']}: {totals['to_revive']} to revive across "
        f"{totals['dives']} dives; digest {report['plan_sha256']}; report -> {out}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point."""
    asyncio.run(_run(parse_args(sys.argv[1:] if argv is None else argv)))


if __name__ == "__main__":
    main()

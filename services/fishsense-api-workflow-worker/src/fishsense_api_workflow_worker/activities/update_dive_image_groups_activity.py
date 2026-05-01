"""Activity to reconcile species labels back into dive frame clusters
(stage 6.1).

Ports `scripts/stage6.1_update_dive_image_groups.ipynb`. Walks the
dive's PREDICTION clusters in order, looks up each entry's
`SpeciesLabel.grouping`, and splits/joins the per-cluster sequences
into LABEL_STUDIO clusters according to the labeler's grouping
choices:

* `grouping == "Part of previous group"` keeps appending to the
  current group, even across PREDICTION cluster boundaries.
* `grouping == "Not part of current group"` flushes the current
  group and starts a new one beginning with that label.
* The first entry of each PREDICTION cluster otherwise starts a new
  group.

LABEL_STUDIO clusters are the input to stage 14 measurement
(`measure_fish_activity`). This activity is the gating step between
"species labels arrived" and "fish lengths can be computed."

Idempotency: the activity refuses to run if any LABEL_STUDIO clusters
already exist for the dive — there is no DELETE on the cluster API,
and a re-run that simply POSTs more clusters would silently
double-count. To re-group after labels change, an operator must
manually remove the existing LABEL_STUDIO clusters first.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.data_source import DataSource
from fishsense_api_sdk.models.dive_frame_cluster import DiveFrameCluster
from fishsense_api_sdk.models.species_label import SpeciesLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import get_fs_client

GROUPING_CONTINUE = "Part of previous group"
GROUPING_BREAK = "Not part of current group"


@dataclass
class UpdateDiveImageGroupsResult:
    """Outcome of one stage-6.1 reconciliation for a single dive.

    `skipped_already_grouped` differentiates "no work to do" (existing
    LABEL_STUDIO clusters present) from "no work possible" (no
    PREDICTION clusters or no species labels) — both report
    `new_clusters_created=0`, but the operator's response differs.
    """

    skipped_already_grouped: bool
    new_clusters_created: int
    species_labels_seen: int


__all__ = [
    "UpdateDiveImageGroupsResult",
    "update_dive_image_groups_activity",
    "regroup_by_species_labels",
]


def regroup_by_species_labels(
    prediction_clusters: Iterable[DiveFrameCluster],
    species_label_by_image_id: dict[int, SpeciesLabel],
) -> List[List[int]]:
    """Apply labelers' grouping choices to PREDICTION cluster image_ids.

    Pure function — extracted so the boundary logic can be unit-tested
    without going through the SDK. Returns one list of image_ids per
    LABEL_STUDIO cluster, in the same order PREDICTION clusters were
    visited. Image_ids missing from `species_label_by_image_id` are
    skipped (their grouping is unknown — the notebook would KeyError;
    skipping is the safer port).
    """
    groups: List[List[int]] = []
    current: List[int] = []
    for cluster in prediction_clusters:
        for idx, image_id in enumerate(cluster.image_ids):
            label = species_label_by_image_id.get(image_id)
            if label is None:
                continue
            starts_new_group = (
                idx == 0 and label.grouping != GROUPING_CONTINUE
            ) or label.grouping == GROUPING_BREAK
            if starts_new_group and current:
                groups.append(current)
                current = []
            current.append(image_id)
    if current:
        groups.append(current)
    return groups


async def _post_clusters(
    fs: Client, dive_id: int, groups: List[List[int]]
) -> None:
    now = datetime.now(timezone.utc)
    for image_ids in groups:
        cluster = DiveFrameCluster(
            id=None,
            dive_id=dive_id,
            image_ids=image_ids,
            data_source=DataSource.LABEL_STUDIO,
            fish_id=None,
            updated_at=now,
        )
        await fs.images.post_cluster(dive_id, cluster)
        activity.heartbeat()


@activity.defn
async def update_dive_image_groups_activity(
    dive_id: int,
) -> UpdateDiveImageGroupsResult:
    """Materialize LABEL_STUDIO clusters for `dive_id` from species labels."""
    async with get_fs_client() as fs:
        existing = (
            await fs.images.get_clusters(dive_id, DataSource.LABEL_STUDIO.value)
            or []
        )
        if existing:
            activity.logger.info(
                "dive_id=%d already has %d LABEL_STUDIO clusters; "
                "skipping (delete them first to re-group)",
                dive_id,
                len(existing),
            )
            return UpdateDiveImageGroupsResult(
                skipped_already_grouped=True,
                new_clusters_created=0,
                species_labels_seen=0,
            )

        prediction_clusters = (
            await fs.images.get_clusters(dive_id, DataSource.PREDICTION.value)
            or []
        )
        species_labels = await fs.labels.get_species_labels(dive_id) or []

        species_label_by_image_id = {
            label.image_id: label
            for label in species_labels
            if label.image_id is not None
        }

        groups = regroup_by_species_labels(
            prediction_clusters, species_label_by_image_id
        )

        if not groups:
            activity.logger.info(
                "dive_id=%d: no LABEL_STUDIO groups to create "
                "(prediction_clusters=%d, species_labels=%d)",
                dive_id,
                len(prediction_clusters),
                len(species_labels),
            )
            return UpdateDiveImageGroupsResult(
                skipped_already_grouped=False,
                new_clusters_created=0,
                species_labels_seen=len(species_labels),
            )

        await _post_clusters(fs, dive_id, groups)

        activity.logger.info(
            "dive_id=%d: created %d LABEL_STUDIO clusters from %d species labels",
            dive_id,
            len(groups),
            len(species_labels),
        )
        return UpdateDiveImageGroupsResult(
            skipped_already_grouped=False,
            new_clusters_created=len(groups),
            species_labels_seen=len(species_labels),
        )

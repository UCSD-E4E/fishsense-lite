"""Stage 1 (dive-frame clustering) activity.

Consumes `(image_id, taken_datetime)` pairs from the api-worker parent
via `fishsense_shared.ClusterDiveFramesInput.images` and returns a
list of clusters of image_ids. The math kernel (HDBSCAN on
timestamps, `min_cluster_size=2`) is unchanged from the notebook port;
only the input/output shapes flipped to make the contract serializable
across worker boundaries (no data-worker-local pydantic types in the
cross-worker DTO).
"""

from __future__ import annotations

from typing import Iterable, List

from sklearn.cluster import HDBSCAN
from temporalio import activity

from fishsense_shared import ClusterDiveFrameImage


@activity.defn
async def cluster_dive_frames(images: Iterable[ClusterDiveFrameImage]) -> List[List[int]]:
    """Cluster a dive's images by their taken_datetime timestamps.

    Returns:
        list[list[int]]: image_ids grouped by temporal cluster.
        HDBSCAN noise points (label -1) are dropped.
    """
    image_list = list(images)
    if not image_list:
        return []

    timestamps = [[img.taken_datetime.timestamp()] for img in image_list]

    activity.logger.info("Clustering %d images", len(image_list))

    db = HDBSCAN(min_cluster_size=2).fit(timestamps)
    labels = db.labels_

    clusters: dict[int, List[int]] = {}
    for label, img in zip(labels, image_list):
        if label == -1:
            continue
        clusters.setdefault(int(label), []).append(img.image_id)

    return list(clusters.values())

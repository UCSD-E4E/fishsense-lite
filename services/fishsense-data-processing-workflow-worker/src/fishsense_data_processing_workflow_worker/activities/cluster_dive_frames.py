"""Stage 1 (dive-frame clustering) activity.

Consumes `(image_id, taken_datetime)` pairs from the api-worker parent
via `fishsense_shared.ClusterDiveFramesInput.images` and returns a
list of clusters of image_ids. The math kernel (HDBSCAN on
timestamps, `min_cluster_size=2`) is unchanged from the notebook port;
only the input/output shapes flipped to make the contract serializable
across worker boundaries (no data-worker-local pydantic types in the
cross-worker DTO).

**Every input frame comes back in exactly one cluster**, and that is the
contract this module exists to keep. The notebook port dropped HDBSCAN noise
points (label -1), which caused two defects with one root cause:

* A dive whose frames are EVENLY SPACED has no density variation for HDBSCAN to
  find, so every point is noise and the activity returned `[]`. Nothing
  persisted, the stage-1 cohort's "has no PREDICTION cluster" gate stayed true,
  and the dive was re-selected hourly forever -- head-of-line blocking every
  higher-id dive, since the selector is `ORDER BY id LIMIT 1`. Prod dive 8
  (2 frames, 2 s apart) was selected 8 times in a row on 2026-09-16, and
  uniform spacing was measured to return all-noise for n = 2..6.
* A dive holding exactly ONE frame made HDBSCAN raise
  (`n_samples=1 while HDBSCAN requires more than one sample`), failing the
  activity rather than returning nothing -- so the child and parent failed too,
  and the dive still never drained. Guarded above.
* Even where clustering worked, noise frames vanished. Prod dive 5 holds 17
  canonical frames and only 16 reached clusters, so one frame had no PREDICTION
  cluster, never entered stage 2, and was invisible to species labelling.

A noise point therefore becomes its own singleton cluster. That is the honest
neutral answer rather than a workaround: a PREDICTION cluster is a *prediction*
which labelers correct in stage 6.1, and "this frame groups with nothing" is
what the data says. The alternative -- one cluster holding every unclustered
frame -- would assert a grouping the data does not support, and would be
actively wrong on a reef dive holding many different fish.
"""

from __future__ import annotations

from typing import Iterable, List

from sklearn.cluster import HDBSCAN
from temporalio import activity

from fishsense_shared import ClusterDiveFrameImage


@activity.defn
async def cluster_dive_frames(
    images: Iterable[ClusterDiveFrameImage],
) -> List[List[int]]:
    """Cluster a dive's images by their taken_datetime timestamps.

    Returns:
        list[list[int]]: image_ids grouped by temporal cluster. Every input
        frame appears in exactly one cluster; a frame HDBSCAN calls noise
        becomes a singleton rather than being dropped. Input order is
        preserved within and across clusters, so the output is deterministic
        for a given input.
    """
    image_list = list(images)
    if not image_list:
        return []

    if len(image_list) == 1:
        # HDBSCAN raises `n_samples=1 while HDBSCAN requires more than one
        # sample`, so a one-frame dive would fail the activity outright rather
        # than merely returning nothing -- the child workflow fails, the parent
        # fails, and the dive still never drains. One frame is one group.
        activity.logger.info("Clustering 1 image: single frame, one cluster")
        return [[image_list[0].image_id]]

    timestamps = [[img.taken_datetime.timestamp()] for img in image_list]

    activity.logger.info("Clustering %d images", len(image_list))

    db = HDBSCAN(min_cluster_size=2).fit(timestamps)
    labels = db.labels_

    clusters: dict[int, List[int]] = {}
    singletons: List[List[int]] = []
    for label, img in zip(labels, image_list):
        if int(label) == -1:
            # Noise: its own cluster. Not keyed into `clusters`, because every
            # noise point shares the label -1 and would otherwise collapse into
            # one bogus group.
            singletons.append([img.image_id])
            continue
        clusters.setdefault(int(label), []).append(img.image_id)

    grouped = list(clusters.values())
    activity.logger.info(
        "Clustered %d images into %d group(s) and %d singleton(s)",
        len(image_list),
        len(grouped),
        len(singletons),
    )
    return grouped + singletons

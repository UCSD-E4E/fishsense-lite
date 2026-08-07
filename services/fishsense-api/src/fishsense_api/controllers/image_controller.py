"""Image Controller for FishSense API."""

import asyncio
import logging
import re
from typing import Any, Dict, List

from fastapi import Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from fishsense_api.database import get_async_session
from fishsense_api.models.data_source import DataSource
from fishsense_api.models.dive_frame_cluster import (
    DiveFrameCluster,
    DiveFrameClusterImageMapping,
    DiveFrameClusterJson,
)
from fishsense_api.models.dive import Dive
from fishsense_api.models.image import Image
from fishsense_api.server import app

logger = logging.getLogger(__name__)


@app.get("/api/v1/images/{image_id}")
async def get_image(
    image_id: int, session: AsyncSession = Depends(get_async_session)
) -> Image | None:
    """Retrieve an image by its ID."""
    logger.debug("Retrieving image with id=%d", image_id)
    query = select(Image).where(Image.id == image_id)

    image = (await session.exec(query)).first()
    if image is None:
        logger.warning("Image with id=%d not found", image_id)
        raise HTTPException(status_code=404, detail="Image not found")
    return image


@app.get("/api/v1/images/checksum/{checksum}")
async def get_image_by_checksum(
    checksum: str, session: AsyncSession = Depends(get_async_session)
) -> Image | None:
    """Retrieve an image by its checksum."""
    logger.debug("Retrieving image with checksum=%s", checksum)
    query = select(Image).where(Image.checksum == checksum)

    image = (await session.exec(query)).first()
    if image is None:
        logger.warning("Image with checksum=%s not found", checksum)
        raise HTTPException(status_code=404, detail="Image not found")
    return image


# Mirrors `max_length=255` on `Image.path`; see `post_dive` for why this is
# checked explicitly rather than left to the driver.
MAX_PATH_LENGTH = 255

# `Image.checksum` is a 32-char lowercase MD5 hexdigest of the whole file --
# `hashlib.md5(<entire file>).hexdigest()`, matching `get_file_checksum` in the
# (now archived) fishsense-data-processing-spider, which is what wrote every
# `image_md5` the current rows were migrated from. A value of any other shape
# means the hashing changed underneath us, and duplicate detection would stop
# matching *silently* -- so it is rejected rather than stored.
_MD5_HEXDIGEST = re.compile(r"^[0-9a-f]{32}$")

# Cap on one `post_checksum_lookup` batch. A dive is a few hundred frames, so
# this is generous; it exists to stop an unbounded IN (...) from a bad caller.
MAX_CHECKSUM_LOOKUP = 1000


@app.post("/api/v1/dives/{dive_id}/images/", status_code=201)
async def post_image(
    dive_id: int,
    image: Image,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create or update an image, keyed on its NAS-relative `path`.

    **Upserts on `path`** for the same reason `post_dive` does: a resumed scan
    re-posts paths that already exist, and a blind `session.merge(id=None)`
    would INSERT and trip the unique index (#347).

    **`is_canonical` is computed here, not by the caller.** The rule comes from
    the original migration (`9e5bc64`): the first row for a given checksum is
    canonical, later duplicates are not. The same physical frames legitimately
    appear under two dive rows -- prod dives 64 and 66 are both
    `082929_FishModels_FSL07` -- and `checksum` is how that is recognised.

    Computing it server-side matters: a client-side "has this checksum been
    seen yet" check races itself, and two concurrent posts would both conclude
    they were first. An explicit `is_canonical` in the body still wins, so an
    operator can promote a re-ingested copy.
    """
    logger.debug("Creating or updating image with path=%s", image.path)

    # Path/checksum checks run before `model_validate` -- see `post_dive`.
    if not image.path:
        raise HTTPException(status_code=422, detail="Image path is required")
    if len(image.path) > MAX_PATH_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Image path exceeds {MAX_PATH_LENGTH} characters "
                f"({len(image.path)}): {image.path}"
            ),
        )
    if not image.checksum or not _MD5_HEXDIGEST.match(image.checksum):
        raise HTTPException(
            status_code=422,
            detail=(
                "checksum must be a 32-character lowercase MD5 hexdigest of the "
                f"whole file; got {image.checksum!r}"
            ),
        )
    if image.taken_datetime is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "taken_datetime is required; stage-1 clustering is pure timestamp "
                "math, so a missing value would corrupt it silently"
            ),
        )

    dive = (await session.exec(select(Dive).where(Dive.id == dive_id))).first()
    if dive is None:
        raise HTTPException(status_code=422, detail=f"Dive {dive_id} does not exist")

    explicit_is_canonical = "is_canonical" in image.model_fields_set

    image = Image.model_validate(jsonable_encoder(image))
    image.dive_id = dive_id

    # Natural-key upsert on `path`.
    if image.id is None:
        existing = (
            await session.exec(select(Image).where(Image.path == image.path))
        ).first()
        if existing is not None:
            image.id = existing.id

    if not explicit_is_canonical:
        # "Is there already a row with this checksum that ISN'T this one?"
        # Excluding self matters: a resumed scan re-posts existing paths, and
        # without the exclusion every re-run would demote a whole dive to
        # non-canonical by colliding with itself.
        duplicate_query = select(Image).where(Image.checksum == image.checksum)
        if image.id is not None:
            duplicate_query = duplicate_query.where(Image.id != image.id)
        duplicate = (await session.exec(duplicate_query)).first()
        image.is_canonical = duplicate is None

    image = await session.merge(image)
    await session.flush()

    image_id = image.id

    return image_id


@app.post("/api/v1/images/checksums/lookup")
async def post_checksum_lookup(
    checksums: List[str],
    session: AsyncSession = Depends(get_async_session),
) -> Dict[str, List[Dict[str, Any]]]:
    """Batch form of `GET /api/v1/images/checksum/{checksum}`.

    Returns `{checksum: [{image_id, dive_id, is_canonical}, ...]}` with an
    **empty list** for checksums that aren't known -- callers computing
    `|new & existing| / |new|` shouldn't have to guard every lookup.

    This backs duplicate-dive detection at ingest time. It replaces the
    approach the legacy spider used -- a whole-dive digest,
    `MD5(STRING_AGG(basename || ':' || image_md5 ORDER BY path))` -- which was
    all-or-nothing (one extra frame made two near-identical folders look
    entirely unrelated), basename-sensitive (a rename broke the match even when
    the bytes were identical), and offered no similarity measure. A set
    operation over content hashes has none of those properties: it is immune to
    filenames and ordering, and it degrades gracefully to a partial overlap.
    """
    unique = list(dict.fromkeys(checksums))
    if len(unique) > MAX_CHECKSUM_LOOKUP:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Too many checksums in one lookup ({len(unique)}); "
                f"maximum is {MAX_CHECKSUM_LOOKUP}"
            ),
        )
    logger.debug("Looking up %d checksums", len(unique))

    result: Dict[str, List[Dict[str, Any]]] = {checksum: [] for checksum in unique}
    if not unique:
        return result

    rows = (
        await session.exec(
            select(Image).where(Image.checksum.in_(unique))  # pylint: disable=no-member
        )
    ).all()
    for row in rows:
        result[row.checksum].append(
            {
                "image_id": row.id,
                "dive_id": row.dive_id,
                "is_canonical": row.is_canonical,
            }
        )

    return result


@app.get("/api/v1/dives/{dive_id}/images/")
async def get_dive_images(
    dive_id: int, session: AsyncSession = Depends(get_async_session)
) -> List[Image] | None:
    """Retrieve all images associated with a specific dive ID."""
    logger.debug("Retrieving images for dive with id=%d", dive_id)
    query = select(Image).where(Image.dive_id == dive_id)

    images = (await session.exec(query)).all()
    if not images:
        logger.warning("Images for dive with id=%d not found", dive_id)
        raise HTTPException(status_code=404, detail="Images not found")
    return images


@app.get("/api/v1/dives/{dive_id}/images/clusters/{data_source}")
async def get_clusters(
    dive_id: int,
    data_source: DataSource,
    session: AsyncSession = Depends(get_async_session),
) -> List[DiveFrameClusterJson] | None:
    """Retrieve all image clusters associated with a specific dive ID."""
    logger.debug(
        "Retrieving image clusters for dive with id=%d and data_source=%s",
        dive_id,
        data_source,
    )
    query = (
        select(DiveFrameCluster)
        .where(DiveFrameCluster.dive_id == dive_id)
        .where(DiveFrameCluster.data_source == data_source)
    )

    clusters = (await session.exec(query)).all()
    cluster_mapping_query = (
        select(DiveFrameClusterImageMapping)
        .join(
            DiveFrameCluster,
            DiveFrameClusterImageMapping.dive_frame_cluster_id == DiveFrameCluster.id,
        )
        .where(DiveFrameCluster.data_source == data_source)
        .where(DiveFrameCluster.dive_id == dive_id)
    )
    cluster_mappings = (await session.exec(cluster_mapping_query)).all()

    cluster_mappings_dict: Dict[int, List[DiveFrameClusterImageMapping]] = {}
    for mappings in cluster_mappings:
        if mappings.dive_frame_cluster_id not in cluster_mappings_dict:
            cluster_mappings_dict[mappings.dive_frame_cluster_id] = []

        cluster_mappings_dict[mappings.dive_frame_cluster_id].append(mappings)

    return [
        DiveFrameClusterJson(
            id=c.id,
            image_ids=[m.image_id for m in cluster_mappings_dict.get(c.id, [])],
            data_source=c.data_source,
            updated_at=c.updated_at,
            dive_id=c.dive_id,
            fish_id=c.fish_id,
        )
        for c in clusters
    ]


@app.post("/api/v1/dives/{dive_id}/images/clusters/", status_code=201)
async def post_cluster(
    dive_id: int,
    dive_frame_cluster: DiveFrameClusterJson,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Create a new image cluster for a specific dive ID."""
    logger.debug("Creating a new image cluster for dive with id=%d", dive_id)
    dive_frame_cluster = DiveFrameClusterJson.model_validate(
        jsonable_encoder(dive_frame_cluster)
    )
    images = (
        await session.exec(
            select(Image).where(
                Image.id.in_(dive_frame_cluster.image_ids)  # pylint: disable=no-member
            )
        )
    ).all()

    dive_frame_cluster = DiveFrameCluster(
        dive_id=dive_id,
        data_source=dive_frame_cluster.data_source,
        updated_at=dive_frame_cluster.updated_at,
        fish_id=dive_frame_cluster.fish_id,
    )
    dive_frame_cluster = await session.merge(dive_frame_cluster)
    await session.flush()  # Ensure ID is populated

    dive_frame_cluster_id = dive_frame_cluster.id  # Access ID to ensure it's loaded

    mappings = []
    for image in images:
        mapping = DiveFrameClusterImageMapping(
            dive_frame_cluster_id=dive_frame_cluster.id, image_id=image.id
        )
        mappings.append(mapping)

    session.add_all(mappings)

    return dive_frame_cluster_id


@app.put("/api/v1/dives/{dive_id}/images/clusters/{dive_frame_cluster_id}")
async def put_cluster(
    dive_id: int,
    dive_frame_cluster_id: int,
    dive_frame_cluster: DiveFrameClusterJson,
    session: AsyncSession = Depends(get_async_session),
) -> int:
    """Update an existing image cluster for a specific dive ID."""
    logger.debug(
        "Updating image cluster with id=%d for dive with id=%d",
        dive_frame_cluster_id,
        dive_id,
    )
    dive_frame_cluster = DiveFrameClusterJson.model_validate(
        jsonable_encoder(dive_frame_cluster)
    )
    images = (
        await session.exec(
            select(Image).where(
                Image.id.in_(dive_frame_cluster.image_ids)  # pylint: disable=no-member
            )
        )
    ).all()

    dive_frame_cluster = DiveFrameCluster(
        id=dive_frame_cluster_id,
        dive_id=dive_id,
        data_source=dive_frame_cluster.data_source,
        updated_at=dive_frame_cluster.updated_at,
        fish_id=dive_frame_cluster.fish_id,
    )
    dive_frame_cluster = await session.merge(dive_frame_cluster)
    await session.flush()  # Ensure ID is populated

    # Clear existing mappings
    mappings_to_delete = await session.exec(
        select(DiveFrameClusterImageMapping).where(
            DiveFrameClusterImageMapping.dive_frame_cluster_id == dive_frame_cluster.id
        )
    )
    await asyncio.gather(
        *[session.delete(mapping) for mapping in mappings_to_delete.all()]
    )

    mappings = []
    for image in images:
        mapping = DiveFrameClusterImageMapping(
            dive_frame_cluster_id=dive_frame_cluster.id, image_id=image.id
        )
        mappings.append(mapping)

    session.add_all(mappings)

    return dive_frame_cluster.id

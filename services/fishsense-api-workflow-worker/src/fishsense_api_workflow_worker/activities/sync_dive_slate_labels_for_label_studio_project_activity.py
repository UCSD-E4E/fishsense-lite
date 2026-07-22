"""Activity to sync dive-slate labels for a Label Studio project (stage 12).

Mirrors the laser + headtail sync pattern, with one slate-specific
twist: the LS image is a composite (PDF panel on the left + photo on
the right), so reference_point and slate_rectangle x-coords need to be
shifted left by the rendered PDF panel width to land in photo-frame
coords. We compute that offset from the slate PDF's intrinsic
points-per-page aspect ratio (no rendering needed) — see
`compute_pdf_panel_width_in_composite`.
"""

import asyncio
import json
from typing import Any, Dict, List, Tuple

import pymupdf
from botocore.exceptions import BotoCoreError, ClientError
from fishsense_api_sdk.client import Client
from fishsense_api_sdk.models.dive_slate_label import DiveSlateLabel
from temporalio import activity

from fishsense_api_workflow_worker.activities.utils import (
    SYNC_CONCURRENCY,
    resolve_annotator_user,
    sync_label_studio_project,
)
from fishsense_api_workflow_worker.object_store import (
    ObjectStoreClient,
    open_object_store_client,
)

__all__ = [
    "sync_dive_slate_labels_for_label_studio_project_activity",
    "SYNC_CONCURRENCY",
]


Point = Tuple[float, float]


def compute_pdf_panel_aspect_ratio(pdf_bytes: bytes) -> float:
    """Read page 0 of a slate PDF and return its width / height in points.

    The composite-image scaling is `original_height / pdf_height`, so the
    rendered panel width is `pdf_width * scale = (pdf_width / pdf_height)
    * original_height`. Because pymupdf reports `page.rect` in
    DPI-independent points (1/72 inch), the width-to-height ratio is the
    only intrinsic we need; DPI cancels out of the offset calculation.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
        page: pymupdf.Page = document.load_page(0)
        rect = page.rect
        return float(rect.width) / float(rect.height)


def compute_pdf_panel_width_in_composite(
    pdf_aspect_ratio: float, original_height: float
) -> float:
    """Pixel width of the PDF panel inside the LS composite image."""
    return pdf_aspect_ratio * float(original_height)


def _shift_x(points: List[Point], dx: float) -> List[Point]:
    return [(x - dx, y) for x, y in points]


def _parse_results(annotation: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the slate annotation fields out of an LS task result list.

    Returns raw composite-frame coordinates (no offset applied) plus
    `original_height`. The caller applies the panel-width shift once it
    has fetched the slate PDF.
    """
    results = annotation.get("result") or []

    upside_down: bool | None = None
    upside_down_results = [r for r in results if r["from_name"] == "upside_down"]
    if upside_down_results:
        choices = upside_down_results[0]["value"].get("choices") or []
        if choices:
            upside_down = choices[0] == "Slate upside down"

    reference_results = [r for r in results if r["from_name"] == "reference_points"]
    reference_points: List[Point] = [
        (
            r["value"]["x"] / 100.0 * r["original_width"],
            r["value"]["y"] / 100.0 * r["original_height"],
        )
        for r in reference_results
    ]

    slate_results = [r for r in results if r["from_name"] == "slate"]
    slate_rectangle: List[Point] | None = None
    if slate_results:
        sr = slate_results[0]
        ow = sr["original_width"]
        oh = sr["original_height"]
        slate_rectangle = [
            (sr["value"]["x"] / 100.0 * ow, sr["value"]["y"] / 100.0 * oh),
            (
                (sr["value"]["x"] + sr["value"]["width"]) / 100.0 * ow,
                (sr["value"]["y"] + sr["value"]["height"]) / 100.0 * oh,
            ),
        ]

    skipped_results = [r for r in results if r["from_name"] == "skipped_points"]
    skipped_points: List[int] | None = None
    if skipped_results:
        text = skipped_results[0]["value"].get("text") or []
        # Notebook stored 0-based indices; LS shows 1-based to humans.
        skipped_points = [int(p) - 1 for p in text]

    original_height: float | None = None
    for r in results:
        if "original_height" in r:
            original_height = float(r["original_height"])
            break

    return {
        "upside_down": upside_down,
        "reference_points": reference_points,
        "slate_rectangle": slate_rectangle,
        "skipped_points": skipped_points,
        "original_height": original_height,
    }


async def _slate_id_for_image(
    fs: Client, image_id: int, image_to_slate: Dict[int, int | None]
) -> int | None:
    """Resolve `image_id -> dive_slate_id` with per-activity memoization."""
    if image_id in image_to_slate:
        return image_to_slate[image_id]

    image = await fs.images.get(image_id=image_id)
    if image is None or image.dive_id is None:
        image_to_slate[image_id] = None
        return None

    dive = await fs.dives.get(dive_id=image.dive_id)
    slate_id = getattr(dive, "dive_slate_id", None) if dive is not None else None
    image_to_slate[image_id] = slate_id
    return slate_id


async def _aspect_ratio_for_slate(
    exchange: ObjectStoreClient,
    slate_id: int,
    aspect_cache: Dict[int, float],
) -> float | None:
    """Fetch the slate PDF (once per activity) from Garage and return
    width/height in points."""
    if slate_id in aspect_cache:
        return aspect_cache[slate_id]

    try:
        pdf_bytes = await exchange.download_slate_pdf(slate_id)
    except (ClientError, BotoCoreError) as e:
        # Missing/unreadable slate PDF in Garage (botocore ClientError or
        # transport error). Skip the panel-width offset for this label
        # rather than failing the whole sync. Anything else (e.g. a
        # programming error) propagates and fails the activity.
        activity.logger.warning(
            "Could not fetch slate PDF for slate_id=%d (%s); "
            "skipping panel-width offset for this label",
            slate_id,
            e,
        )
        return None

    aspect = await asyncio.to_thread(
        compute_pdf_panel_aspect_ratio, pdf_bytes
    )
    aspect_cache[slate_id] = aspect
    return aspect


async def _update_slate_label(
    fs: Client,
    task: Any,
    *,
    exchange: ObjectStoreClient,
    aspect_cache: Dict[int, float],
    image_to_slate: Dict[int, int | None],
) -> None:
    slate_label: DiveSlateLabel | None = await fs.labels.get_dive_slate_label(
        label_studio_id=task.id
    )
    if slate_label is None:
        return

    # Attribution is best-effort: hosted LS returns `annotators` as
    # dicts rather than ints, and mis-handling that used to 422 and
    # kill the whole project's sync. See resolve_annotator_user.
    user = await resolve_annotator_user(fs, task)
    if user is not None:
        slate_label.user_id = user.id

    slate_label.label_studio_json = json.loads(task.json())
    slate_label.completed = task.is_labeled
    slate_label.updated_at = task.updated_at

    if task.annotations:
        annotation = task.annotations[0]
        parsed = _parse_results(annotation)

        if parsed["upside_down"] is not None:
            slate_label.upside_down = parsed["upside_down"]

        if parsed["skipped_points"] is not None:
            slate_label.skipped_points = parsed["skipped_points"]

        panel_width = 0.0
        original_height = parsed["original_height"]
        if (
            (parsed["reference_points"] or parsed["slate_rectangle"])
            and original_height is not None
            and slate_label.image_id is not None
        ):
            slate_id = await _slate_id_for_image(
                fs, slate_label.image_id, image_to_slate
            )
            if slate_id is not None:
                aspect = await _aspect_ratio_for_slate(
                    exchange, slate_id, aspect_cache
                )
                if aspect is not None:
                    panel_width = compute_pdf_panel_width_in_composite(
                        aspect, original_height
                    )

        if parsed["reference_points"]:
            slate_label.reference_points = _shift_x(
                parsed["reference_points"], panel_width
            )

        if parsed["slate_rectangle"] is not None:
            slate_label.slate_rectangle = _shift_x(
                parsed["slate_rectangle"], panel_width
            )

    await fs.labels.put_dive_slate_label(slate_label.image_id, slate_label)


@activity.defn
async def sync_dive_slate_labels_for_label_studio_project_activity(project_id: int):
    """Activity to sync dive-slate labels for a Label Studio project."""
    aspect_cache: Dict[int, float] = {}
    image_to_slate: Dict[int, int | None] = {}

    exchange = open_object_store_client()

    async def _update(fs: Client, task: Any) -> None:
        await _update_slate_label(
            fs,
            task,
            exchange=exchange,
            aspect_cache=aspect_cache,
            image_to_slate=image_to_slate,
        )

    await sync_label_studio_project(project_id, _update, kind="dive_slate")
